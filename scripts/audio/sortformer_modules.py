# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import math
from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class StreamingSortformerState:
    """
    This class creates a class instance that will be used to store the state of the
    streaming Sortformer model.

    Attributes:
        spkcache (torch.Tensor): Speaker cache to store embeddings from start
        spkcache_lengths (torch.Tensor): Lengths of the speaker cache
        spkcache_preds (torch.Tensor): The predictions for the speaker cache parts
        fifo (torch.Tensor): FIFO queue to save the embedding from the latest chunks
        fifo_lengths (torch.Tensor): Lengths of the FIFO queue
    """

    spkcache = None  # Speaker cache to store embeddings from start
    spkcache_lengths = None
    spkcache_preds = None  # speaker cache predictions
    fifo = None  # to save the embedding from the latest chunks
    fifo_lengths = None

    # Cached lengths as Python ints to avoid .item() GPU->CPU sync
    spkcache_len_cached: int = 0
    fifo_len_cached: int = 0

    # Double-buffer for async compression
    spkcache_pending = None  # Pending buffer being written by async compression
    spkcache_preds_pending = None
    spkcache_len_pending: int = 0
    compression_event = None  # CUDA event to track completion
    has_pending_compression: bool = False

    # Retain tensors used by async compression until sync
    # This MUST be per-state, not per-module, to avoid use-after-free when
    # different requests process different states concurrently
    pending_compression_tensors = None


class SortformerModules:
    """
    A class including auxiliary functions for Sortformer models.
    This class contains and will contain the following functions that performs streaming features,
    and any neural layers that are not included in the NeMo neural modules
    (e.g. Transformer, Fast-Conformer).
    """

    def init_weights(self, m):
        """Init weights for linear layers."""
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def __init__(
        self,
        num_spks: int = 4,
        dropout_rate: float = 0.5,
        fc_d_model: int = 512,
        tf_d_model: int = 192,
        subsampling_factor: int = 8,
        spkcache_len: int = 188,
        fifo_len: int = 0,
        chunk_len: int = 376,
        spkcache_refresh_rate: int = 1,
        chunk_left_context: int = 1,
        chunk_right_context: int = 1,
        spkcache_sil_frames_per_spk: int = 3,
        causal_attn_rate: float = 0,
        causal_attn_rc: int = 7,
        scores_add_rnd: float = 0,
        pred_score_threshold: float = 0.25,
        max_index: int = 99999,
        scores_boost_latest: float = 0.05,
        sil_threshold: float = 0.2,
        strong_boost_rate: float = 0.75,
        weak_boost_rate: float = 1.5,
        min_pos_scores_rate: float = 0.5,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        # General params
        self.subsampling_factor = subsampling_factor
        self.fc_d_model = fc_d_model
        self.tf_d_model = tf_d_model
        self.hidden_size = tf_d_model
        self.n_spk: int = num_spks
        self.hidden_to_spks = nn.Linear(2 * self.hidden_size, self.n_spk)
        self.first_hidden_to_hidden = nn.Linear(self.hidden_size, self.hidden_size)
        self.single_hidden_to_spks = nn.Linear(self.hidden_size, self.n_spk)
        self.dropout = nn.Dropout(dropout_rate)
        self.encoder_proj = nn.Linear(self.fc_d_model, self.tf_d_model)
        self.log = False

        # Streaming-related params
        self.spkcache_len = spkcache_len
        self.fifo_len = fifo_len
        self.chunk_len = chunk_len
        self.chunk_left_context = chunk_left_context
        self.chunk_right_context = chunk_right_context
        self.spkcache_sil_frames_per_spk = spkcache_sil_frames_per_spk
        self.spkcache_refresh_rate = spkcache_refresh_rate
        self.causal_attn_rate = causal_attn_rate
        self.causal_attn_rc = causal_attn_rc
        self.scores_add_rnd = scores_add_rnd
        self.max_index = max_index
        self.pred_score_threshold = pred_score_threshold
        self.scores_boost_latest = scores_boost_latest
        self.sil_threshold = sil_threshold
        self.strong_boost_rate = strong_boost_rate
        self.weak_boost_rate = weak_boost_rate
        self.min_pos_scores_rate = min_pos_scores_rate

        self.dtype = dtype

        # Compression stream for async execution
        self._compression_stream = None  # Created lazily on first use

        # NOTE: Pending compression tensors are now stored per-state (not per-module)
        # to avoid use-after-free when different concurrent requests process different states

        # DEBUG: Set to True to disable async compression and force sync (helps isolate race conditions)
        self._force_sync_compression = False  # TEMP: Testing if variance is from async

    def length_to_mask(self, lengths, max_length: int):
        """
        Convert length values to encoder mask input tensor

        Args:
            lengths (torch.Tensor): Tensor containing lengths of sequences
            max_length (int): maximum sequence length

        Returns:
            mask (torch.Tensor): Tensor of shape (batch_size, max_len) containing 0's
                                 in the padded region and 1's elsewhere
        """
        batch_size = lengths.shape[0]
        arange = torch.arange(max_length, device=lengths.device)
        mask = arange.expand(batch_size, max_length) < lengths.unsqueeze(1)
        return mask

    def streaming_feat_loader(
        self, feat_seq, feat_seq_length, feat_seq_offset
    ) -> Tuple[int, torch.Tensor, torch.Tensor, int, int]:
        """
        Load a chunk of feature sequence for streaming inference.

        Args:
            feat_seq (torch.Tensor): Tensor containing feature sequence
                Shape: (batch_size, feat_dim, feat frame count)
            feat_seq_length (torch.Tensor): Tensor containing feature sequence lengths
                Shape: (batch_size,)
            feat_seq_offset (torch.Tensor): Tensor containing feature sequence offsets
                Shape: (batch_size,)

        Returns:
            chunk_idx (int): Index of the current chunk
            chunk_feat_seq (torch.Tensor): Tensor containing the chunk of feature sequence
                Shape: (batch_size, diar frame count, feat_dim)
            feat_lengths (torch.Tensor): Tensor containing lengths of the chunk of feature sequence
                Shape: (batch_size,)
        """
        feat_len = feat_seq.shape[2]
        num_chunks = math.ceil(feat_len / (self.chunk_len * self.subsampling_factor))
        if self.log:
            logging.info(
                f"feat_len={feat_len}, num_chunks={num_chunks}, "
                f"feat_seq_length={feat_seq_length}, feat_seq_offset={feat_seq_offset}"
            )

        stt_feat, end_feat, chunk_idx = 0, 0, 0
        while end_feat < feat_len:
            left_offset = min(self.chunk_left_context * self.subsampling_factor, stt_feat)
            end_feat = min(stt_feat + self.chunk_len * self.subsampling_factor, feat_len)
            right_offset = min(self.chunk_right_context * self.subsampling_factor, feat_len - end_feat)
            chunk_feat_seq = feat_seq[:, :, stt_feat - left_offset : end_feat + right_offset]
            feat_lengths = (feat_seq_length + feat_seq_offset - stt_feat + left_offset).clamp(
                0, chunk_feat_seq.shape[2]
            )
            feat_lengths = feat_lengths * (feat_seq_offset < end_feat)
            stt_feat = end_feat
            chunk_feat_seq_t = torch.transpose(chunk_feat_seq, 1, 2)
            if self.log:
                logging.info(
                    f"chunk_idx: {chunk_idx}, "
                    f"chunk_feat_seq_t shape: {chunk_feat_seq_t.shape}, "
                    f"chunk_feat_lengths: {feat_lengths}"
                )
            yield chunk_idx, chunk_feat_seq_t, feat_lengths, left_offset, right_offset
            chunk_idx += 1

    def forward_speaker_sigmoids(self, hidden_out):
        """
        The final layer that outputs speaker probabilities using the Sigmoid activation function.

        Args:
            hidden_out (torch.Tensor): Tensor containing hidden states from the encoder
                Shape: (batch_size, n_frames, hidden_dim)

        Returns:
            preds (torch.Tensor): Tensor containing speaker probabilities computed using
                the Sigmoid activation function
                Shape: (batch_size, n_frames, n_spk)
        """
        hidden_out = self.dropout(F.relu(hidden_out))
        hidden_out = self.first_hidden_to_hidden(hidden_out)
        hidden_out = self.dropout(F.relu(hidden_out))
        spk_preds = self.single_hidden_to_spks(hidden_out)
        preds = nn.Sigmoid()(spk_preds)
        return preds

    def concat_embs(
        self,
        list_of_tensors=List[torch.Tensor],
        return_lengths: bool = False,
        dim: int = 1,
        device: torch.device = None,
    ):
        """
        Concatenate a list of tensors along the specified dimension.

        Args:
            list_of_tensors (List[torch.Tensor]): List of tensors to concatenate
            return_lengths (bool): Whether to return lengths of the concatenated tensors
            dim (int): Concatenation axis
            device (torch.device): device to use for tensor operations

        Returns:
            embs (torch.Tensor): concatenated tensor
        """
        embs = torch.cat(list_of_tensors, dim=dim).to(device)
        lengths = torch.tensor(embs.shape[1]).repeat(embs.shape[0]).to(device)
        if return_lengths:
            return embs, lengths
        else:
            return embs

    def init_memory(self, batch_size, d_model=192, device=None):
        return torch.zeros(batch_size, 0, d_model).to(device)

    def init_streaming_state(self, device=None):
        """
        Initializes StreamingSortformerState with empty tensors or zero-valued tensors.

        Args:
            device (torch.device): Device for tensors in streaming state

        Returns:
            streaming_state (SortformerStreamingState): initialized streaming state
        """
        streaming_state = StreamingSortformerState()
        streaming_state.spkcache = torch.zeros(
            (1, self.spkcache_len, self.fc_d_model), device=device, dtype=self.dtype
        )
        streaming_state.spkcache_preds = torch.full(
            (1, self.spkcache_len, self.n_spk), -1.0, device=device, dtype=self.dtype
        )
        streaming_state.spkcache_lengths = torch.zeros((1,), dtype=torch.long, device=device)
        streaming_state.fifo = torch.zeros((1, self.fifo_len, self.fc_d_model), device=device, dtype=self.dtype)
        streaming_state.fifo_lengths = torch.zeros((1,), dtype=torch.long, device=device)

        # Initialize cached lengths (avoids GPU->CPU sync via .item())
        streaming_state.spkcache_len_cached = 0
        streaming_state.fifo_len_cached = 0

        # Initialize pending buffers for async compression (double-buffer)
        # Use same -1.0 sentinel for preds_pending as spkcache_preds for consistency after swaps
        streaming_state.spkcache_pending = torch.zeros(
            (1, self.spkcache_len, self.fc_d_model), device=device, dtype=self.dtype
        )
        streaming_state.spkcache_preds_pending = torch.full(
            (1, self.spkcache_len, self.n_spk), -1.0, device=device, dtype=self.dtype
        )
        streaming_state.spkcache_len_pending = 0
        streaming_state.compression_event = None
        streaming_state.has_pending_compression = False

        return streaming_state

    def sync_pending_compression(self, state: StreamingSortformerState):
        """
        Synchronize pending async compression and swap buffers.
        Call this before using state.spkcache for TRT inference.

        Args:
            state: StreamingSortformerState that may have pending compression

        Returns:
            True if sync was needed, False if no pending compression
        """
        if not state.has_pending_compression:
            return False

        # Wait for compression to complete
        if state.compression_event is not None:
            state.compression_event.synchronize()

        # Swap buffers: pending becomes current
        state.spkcache, state.spkcache_pending = state.spkcache_pending, state.spkcache
        state.spkcache_preds, state.spkcache_preds_pending = state.spkcache_preds_pending, state.spkcache_preds
        state.spkcache_len_cached = state.spkcache_len_pending

        # Clear pending state
        state.has_pending_compression = False
        state.compression_event = None

        return True

    def sync_pending_compression_batched(self, batch_states: List[StreamingSortformerState]):
        """
        Sync pending compression for a batch of states.
        Only syncs states that have pending compression.

        Args:
            batch_states: List of states to sync

        Returns:
            Number of states that were synced
        """
        synced = 0
        # First, collect all events that need syncing
        events_to_sync = []
        states_to_swap = []

        for state in batch_states:
            if state.has_pending_compression and state.compression_event is not None:
                events_to_sync.append(state.compression_event)
                states_to_swap.append(state)

        # Sync all events (they may be the same event for batched compression)
        seen_events = set()
        for event in events_to_sync:
            event_id = id(event)
            if event_id not in seen_events:
                event.synchronize()
                seen_events.add(event_id)

        # Swap buffers for all states that had pending compression
        for state in states_to_swap:
            state.spkcache, state.spkcache_pending = state.spkcache_pending, state.spkcache
            state.spkcache_preds, state.spkcache_preds_pending = state.spkcache_preds_pending, state.spkcache_preds
            state.spkcache_len_cached = state.spkcache_len_pending
            state.has_pending_compression = False
            state.compression_event = None
            # Release per-state retained tensors now that sync is complete
            state.pending_compression_tensors = None
            synced += 1

        return synced

    def apply_mask_to_preds(self, spkcache_fifo_chunk_preds, spkcache_fifo_chunk_fc_encoder_lengths):
        """
        Applies mask to speaker cache and FIFO queue to ensure that only valid frames are
        considered for predictions from the model.

        Args:
            spkcache_fifo_chunk_preds (torch.Tensor): Speaker predictions of the chunk
            spkcache_fifo_chunk_fc_encoder_lengths (torch.Tensor): Lengths of current chunk in
                                                                   the Fast-Conformer encoder

        Returns:
            spkcache_fifo_chunk_preds (torch.Tensor): Speaker predictions of the chunk with valid frames only
        """
        batch_size, n_frames, n_spk = spkcache_fifo_chunk_preds.shape
        preds_mask = torch.arange(n_frames, device=spkcache_fifo_chunk_preds.device).view(1, -1, 1)
        preds_mask = preds_mask.expand(batch_size, -1, n_spk) < spkcache_fifo_chunk_fc_encoder_lengths.view(-1, 1, 1)
        preds_mask = preds_mask.expand(-1, n_frames, n_spk)
        spkcache_fifo_chunk_preds = torch.where(
            preds_mask, spkcache_fifo_chunk_preds, torch.zeros_like(spkcache_fifo_chunk_preds)
        )
        return spkcache_fifo_chunk_preds

    # Profiling state for batched update
    _batched_profile_enabled = False
    _batched_profile_count = 0
    _batched_profile_interval = 10  # Print stats every N calls
    _batched_profile_max_samples = 100  # Keep last N samples in memory
    _batched_profile_times = None
    _batched_profile_path_counts = None  # Track path usage
    _batched_profile_detail = None  # Store detailed breakdown for interval printing

    def streaming_update_batched(
        self,
        batch_states: List,
        chunk_embs: torch.Tensor,
        chunk_emb_lengths,  # Can be torch.Tensor OR List[int] for optimization
        preds: torch.Tensor,
        lc_list: List[int],
        rc_list: List[int],
        end_flags: List[int] = None,
    ):
        """
        Fully optimized batched state update.

        Key optimizations:
        - Pre-allocates all output tensors upfront
        - Uses vectorized operations for prediction extraction
        - Batches compression for all sequences that need it
        - Minimizes GPU kernel launches
        - Skips state updates for ending streams (end_flags=1)
        - Accepts chunk_emb_lengths as list to avoid GPU sync inside this function

        Args:
            batch_states: List of B StreamingSortformerState objects
            chunk_embs: (B, max_chunk_len, emb_dim) embeddings from encoder
            chunk_emb_lengths: (B,) actual lengths per sequence (tensor or list)
            preds: (B, max_pred_len, n_spk) predictions for [spkcache + fifo + chunk]
            lc_list: List of B left context offsets
            rc_list: List of B right context offsets
            end_flags: List of B end flags (1=ending, 0=continuing). If provided,
                       state updates are skipped for ending streams.

        Returns:
            batch_states: Updated states (only for non-ending streams)
            chunk_preds: (B, max_chunk_len, n_spk) predictions for chunk parts
            updated_indices: List of batch indices that had state updates (None if end_flags not provided)
        """
        B = len(batch_states)
        if B == 0:
            return batch_states, torch.empty(0, 0, self.n_spk, device=preds.device, dtype=self.dtype)

        # Profiling setup - use CUDA events for accurate GPU timing
        if self._batched_profile_enabled:
            if self._batched_profile_times is None:
                self._batched_profile_times = {
                    "1_lengths": [],
                    "2_stack": [],
                    "3_alloc": [],
                    "4_gather": [],
                    "5_fifo_loop": [],
                    "6_compress": [],
                    "7_writeback": [],
                    "total": [],
                }
            if self._batched_profile_path_counts is None:
                self._batched_profile_path_counts = {
                    "fastest": 0,
                    "fast_full": 0,
                    "fast_varfifo": 0,
                    "semi_fast": 0,
                    "semi_fast_groups": [],
                    "semi_fast_savings": [],
                }
            evt_start = torch.cuda.Event(enable_timing=True)
            evt_after_lengths = torch.cuda.Event(enable_timing=True)
            evt_after_stack = torch.cuda.Event(enable_timing=True)
            evt_after_alloc = torch.cuda.Event(enable_timing=True)
            evt_after_gather = torch.cuda.Event(enable_timing=True)
            evt_after_fifo = torch.cuda.Event(enable_timing=True)
            evt_after_compress = torch.cuda.Event(enable_timing=True)
            evt_end = torch.cuda.Event(enable_timing=True)
            evt_start.record()

        device = preds.device
        emb_dim = self.fc_d_model
        n_spk = self.n_spk
        max_spkcache = self.spkcache_len
        max_fifo = self.fifo_len

        # ========== PHASE 1: Gather all lengths (Python, no GPU ops) ==========
        spk_lens = [s.spkcache_len_cached for s in batch_states]
        fifo_lens = [s.fifo_len_cached for s in batch_states]

        # Get chunk lengths - accept both tensor and list to avoid sync inside this function
        # If caller already converted to list (recommended), no sync here
        # Actual chunk length = total_length - left_context - right_context
        if isinstance(chunk_emb_lengths, list):
            chunk_lens_raw = chunk_emb_lengths
        else:
            chunk_lens_raw = chunk_emb_lengths.tolist()
        chunk_lens = [max(0, chunk_lens_raw[i] - lc_list[i] - rc_list[i]) for i in range(B)]
        max_chunk_len = max(chunk_lens) if chunk_lens else 0

        # Compute pop_out_len and new lengths for each sequence (pure Python)
        pop_out_lens = []
        new_fifo_lens = []
        new_spk_lens = []
        needs_compression = []

        for i in range(B):
            fifo_len = fifo_lens[i]
            chunk_len = chunk_lens[i]
            spk_len = spk_lens[i]

            new_fifo_len = fifo_len + chunk_len
            pop_out_len = 0
            new_spk_len = spk_len

            if new_fifo_len > max_fifo:
                if self.fifo_len == 0:
                    pop_out_len = chunk_len
                elif self.spkcache_refresh_rate == 0:
                    pop_out_len = self.fifo_len
                else:
                    pop_out_len = min(max(self.spkcache_refresh_rate, chunk_len), self.fifo_len)
                pop_out_len = min(pop_out_len, new_fifo_len)
                new_fifo_len -= pop_out_len
                new_spk_len = spk_len + pop_out_len

            pop_out_lens.append(pop_out_len)
            new_fifo_lens.append(new_fifo_len)
            new_spk_lens.append(min(new_spk_len, max_spkcache))
            needs_compression.append(new_spk_len > max_spkcache)

        if self._batched_profile_enabled:
            evt_after_lengths.record()

        # ========== PHASE 2: Stack state tensors (3 GPU ops total) ==========
        # Use torch.stack for efficiency
        spkcache_batch = torch.stack([s.spkcache[0] for s in batch_states], dim=0)
        spkcache_preds_batch = torch.stack([s.spkcache_preds[0] for s in batch_states], dim=0)
        fifo_batch = torch.stack([s.fifo[0] for s in batch_states], dim=0)

        if self._batched_profile_enabled:
            evt_after_stack.record()

        # ========== PHASE 3: Pre-allocate output tensors (3 GPU ops - avoid clone) ==========
        chunk_preds_out = torch.zeros((B, max_chunk_len, n_spk), device=device, dtype=self.dtype)
        new_fifo_batch = torch.zeros_like(fifo_batch)
        # Don't clone - we'll copy only what we need in the update loop
        new_spkcache_batch = torch.zeros_like(spkcache_batch)
        new_spkcache_preds_batch = torch.zeros_like(spkcache_preds_batch)

        if self._batched_profile_enabled:
            evt_after_alloc.record()

        # ========== PHASE 4: Extract chunk predictions (simple loop - avoids tensor allocation overhead) ==========
        # Direct slicing is faster than gather with all its temporary tensor allocations
        for i in range(B):
            chunk_len = chunk_lens[i]
            if chunk_len > 0:
                start = spk_lens[i] + fifo_lens[i] + lc_list[i]
                chunk_preds_out[i, :chunk_len, :] = preds[i, start : start + chunk_len, :]

        if self._batched_profile_enabled:
            evt_after_gather.record()

        # ========== PHASE 4.5: Filter out ending streams (skip state updates) ==========
        # For streams with end_flag=1, we only need chunk_preds - state will be cleared anyway
        if end_flags is not None:
            active_indices = [i for i in range(B) if end_flags[i] != 1]
            if not active_indices:
                # All streams are ending - return early, skip all state update work
                if self._batched_profile_enabled:
                    evt_after_fifo.record()
                    evt_after_compress.record()
                    evt_end.record()
                    torch.cuda.synchronize()
                    # Record zeros for skipped phases
                    self._batched_profile_times["5_fifo_loop"].append(0.0)
                    self._batched_profile_times["6_compress"].append(0.0)
                    self._batched_profile_times["7_writeback"].append(0.0)
                    self._batched_profile_times["total"].append(evt_start.elapsed_time(evt_end))
                return batch_states, chunk_preds_out, []

            # Filter to only active (non-ending) streams
            B_active = len(active_indices)

            # Re-map all per-item lists to active subset
            spk_lens = [spk_lens[i] for i in active_indices]
            fifo_lens = [fifo_lens[i] for i in active_indices]
            chunk_lens = [chunk_lens[i] for i in active_indices]
            lc_list = [lc_list[i] for i in active_indices]
            pop_out_lens = [pop_out_lens[i] for i in active_indices]
            new_fifo_lens = [new_fifo_lens[i] for i in active_indices]
            new_spk_lens = [new_spk_lens[i] for i in active_indices]
            needs_compression = [needs_compression[i] for i in active_indices]

            # Filter batch tensors using index_select (single GPU op)
            idx_tensor = torch.tensor(active_indices, device=device, dtype=torch.long)
            spkcache_batch = spkcache_batch.index_select(0, idx_tensor)
            spkcache_preds_batch = spkcache_preds_batch.index_select(0, idx_tensor)
            fifo_batch = fifo_batch.index_select(0, idx_tensor)
            chunk_embs = chunk_embs.index_select(0, idx_tensor)
            preds = preds.index_select(0, idx_tensor)

            # Re-allocate output tensors for smaller batch
            new_fifo_batch = torch.zeros((B_active, max_fifo, emb_dim), device=device, dtype=self.dtype)
            new_spkcache_batch = torch.zeros((B_active, max_spkcache, emb_dim), device=device, dtype=self.dtype)
            new_spkcache_preds_batch = torch.zeros((B_active, max_spkcache, n_spk), device=device, dtype=self.dtype)

            # Update B for remaining processing
            B = B_active
            updated_indices = active_indices
        else:
            updated_indices = None

        # ========== PHASE 5 & 6: Build combined FIFO, extract pop_out, update spkcache ==========
        # OPTIMIZED v2: Use batched operations where possible, minimize Python loop overhead

        max_combined_fifo = max_fifo + max_chunk_len

        # Fast path: If all lc, fifo_lens, and chunk_lens are uniform across batch,
        # we can do fully batched concat
        uniform_lc = len(set(lc_list)) == 1
        uniform_fifo = len(set(fifo_lens)) == 1
        uniform_spk = len(set(spk_lens)) == 1
        uniform_chunk = len(set(chunk_lens)) == 1

        if uniform_lc and uniform_fifo and uniform_chunk:
            # Fast path: fully vectorized
            fifo_len = fifo_lens[0]
            chunk_len = chunk_lens[0]
            lc = lc_list[0]

            combined_fifo_batch = torch.zeros((B, max_combined_fifo, emb_dim), device=device, dtype=self.dtype)
            combined_fifo_preds_batch = torch.zeros((B, max_combined_fifo, n_spk), device=device, dtype=self.dtype)

            # Single batched copy for FIFO
            if fifo_len > 0:
                combined_fifo_batch[:, :fifo_len, :] = fifo_batch[:, :fifo_len, :]
                # For preds, spk_lens may vary, so we still need a loop for preds
                for i in range(B):
                    start = spk_lens[i]
                    combined_fifo_preds_batch[i, :fifo_len, :] = preds[i, start : start + fifo_len, :]

            # Single batched copy for chunk (with lc offset)
            if chunk_len > 0:
                chunk_embs_seq_len = chunk_embs.shape[1]
                chunk_end = min(lc + chunk_len, chunk_embs_seq_len)
                chunk_actual_len = chunk_end - lc
                if chunk_actual_len > 0:
                    combined_fifo_batch[:, fifo_len : fifo_len + chunk_actual_len, :] = chunk_embs[:, lc:chunk_end, :]
                for i in range(B):
                    start = spk_lens[i] + fifo_len + lc
                    combined_fifo_preds_batch[i, fifo_len : fifo_len + chunk_actual_len, :] = preds[
                        i, start : start + chunk_actual_len, :
                    ]
        else:
            # Slow path: variable lengths
            combined_fifo_batch = torch.zeros((B, max_combined_fifo, emb_dim), device=device, dtype=self.dtype)
            combined_fifo_preds_batch = torch.zeros((B, max_combined_fifo, n_spk), device=device, dtype=self.dtype)

            for i in range(B):
                fifo_len = fifo_lens[i]
                chunk_len = chunk_lens[i]
                lc = lc_list[i]
                spk_len = spk_lens[i]

                if fifo_len > 0:
                    combined_fifo_batch[i, :fifo_len, :] = fifo_batch[i, :fifo_len, :]
                    start = spk_len
                    combined_fifo_preds_batch[i, :fifo_len, :] = preds[i, start : start + fifo_len, :]

                if chunk_len > 0:
                    chunk_embs_seq_len = chunk_embs.shape[1]
                    chunk_end = min(lc + chunk_len, chunk_embs_seq_len)
                    chunk_actual_len = chunk_end - lc
                    if chunk_actual_len > 0:
                        combined_fifo_batch[i, fifo_len : fifo_len + chunk_actual_len, :] = chunk_embs[
                            i, lc:chunk_end, :
                        ]
                    start = spk_len + fifo_len + lc
                    combined_fifo_preds_batch[i, fifo_len : fifo_len + chunk_actual_len, :] = preds[
                        i, start : start + chunk_actual_len, :
                    ]

        # Process pop_out and update states
        compress_inputs_embs = []
        compress_inputs_preds = []
        compress_indices = []

        # Check if we can use fast path for pop_out processing
        max_pop = max(pop_out_lens) if pop_out_lens else 0
        uniform_pop = len(set(pop_out_lens)) == 1
        uniform_spk = len(set(spk_lens)) == 1
        uniform_fifo = len(set(new_fifo_lens)) == 1

        if max_pop == 0 and uniform_spk and uniform_fifo:
            # Fastest path: no pop_out at all, just copy FIFO and keep spkcache as-is
            if self._batched_profile_enabled:
                self._batched_profile_path_counts["fastest"] += 1
            new_fifo_len = new_fifo_lens[0]
            spk_len = spk_lens[0]
            new_fifo_batch[:, :new_fifo_len, :] = combined_fifo_batch[:, :new_fifo_len, :]
            new_spkcache_batch[:, :spk_len, :] = spkcache_batch[:, :spk_len, :]
            new_spkcache_preds_batch[:, :spk_len, :] = spkcache_preds_batch[:, :spk_len, :]
        elif uniform_pop and uniform_spk and max_pop > 0:
            # Fast path: uniform pop_out and spk lengths (can batch spkcache operations)
            pop_out_len = pop_out_lens[0]
            spk_len = spk_lens[0]

            # FIFO extraction - batched if uniform, per-item otherwise
            if uniform_fifo:
                if self._batched_profile_enabled:
                    self._batched_profile_path_counts["fast_full"] += 1
                new_fifo_len = new_fifo_lens[0]
                new_fifo_batch[:, :new_fifo_len, :] = combined_fifo_batch[
                    :, pop_out_len : pop_out_len + new_fifo_len, :
                ]
            else:
                if self._batched_profile_enabled:
                    self._batched_profile_path_counts["fast_varfifo"] += 1
                for i in range(B):
                    new_fifo_len = new_fifo_lens[i]
                    new_fifo_batch[i, :new_fifo_len, :] = combined_fifo_batch[
                        i, pop_out_len : pop_out_len + new_fifo_len, :
                    ]

            # Batched spkcache operations
            max_extended = max_spkcache + pop_out_len
            extended_spkcache_batch = torch.zeros((B, max_extended, emb_dim), device=device, dtype=self.dtype)
            extended_spkcache_preds_batch = torch.zeros((B, max_extended, n_spk), device=device, dtype=self.dtype)

            extended_spkcache_batch[:, :spk_len, :] = spkcache_batch[:, :spk_len, :]
            extended_spkcache_batch[:, spk_len : spk_len + pop_out_len, :] = combined_fifo_batch[:, :pop_out_len, :]

            # Check for first-time (preds have -1.0 sentinel) - need to use TRT preds
            # This happens when spkcache has been accumulated but never compressed yet
            first_time_mask = spkcache_preds_batch[:, 0, 0] < 0

            # Vectorized: use torch.where instead of per-item loop
            extended_spkcache_preds_batch[:, :spk_len, :] = torch.where(
                first_time_mask.view(B, 1, 1), preds[:, :spk_len, :], spkcache_preds_batch[:, :spk_len, :]
            )

            extended_spkcache_preds_batch[:, spk_len : spk_len + pop_out_len, :] = combined_fifo_preds_batch[
                :, :pop_out_len, :
            ]

            # Process compression needs (vectorized)
            needs_compression_t = torch.tensor(needs_compression, dtype=torch.bool, device=device)
            compress_mask = needs_compression_t
            no_compress_mask = ~needs_compression_t

            # Batched copy for non-compression items (single operation instead of loop)
            new_spk_len = spk_len + pop_out_len
            no_compress_idx = no_compress_mask.nonzero(as_tuple=True)[0]
            if len(no_compress_idx) > 0:
                new_spkcache_batch[no_compress_idx, :new_spk_len, :] = extended_spkcache_batch[
                    no_compress_idx, :new_spk_len, :
                ]
                new_spkcache_preds_batch[no_compress_idx, :new_spk_len, :] = extended_spkcache_preds_batch[
                    no_compress_idx, :new_spk_len, :
                ]

            # Collect compression indices (uniform ext_len in fast path)
            compress_indices = compress_mask.nonzero(as_tuple=True)[0].tolist()
            if compress_indices:
                ext_len = spk_len + pop_out_len
                compress_inputs_embs = [extended_spkcache_batch[i, :ext_len, :] for i in compress_indices]
                compress_inputs_preds = [extended_spkcache_preds_batch[i, :ext_len, :] for i in compress_indices]
        else:
            # Semi-fast path: group items by (pop_out_len, spk_len, new_fifo_len) and batch within groups
            max_extended = max_spkcache + max_pop if max_pop > 0 else max_spkcache
            extended_spkcache_batch = torch.zeros((B, max_extended, emb_dim), device=device, dtype=self.dtype)
            extended_spkcache_preds_batch = torch.zeros((B, max_extended, n_spk), device=device, dtype=self.dtype)

            # Group items by their length tuple for batched processing
            groups = {}
            for i in range(B):
                key = (pop_out_lens[i], spk_lens[i], new_fifo_lens[i])
                if key not in groups:
                    groups[key] = []
                groups[key].append(i)

            if self._batched_profile_enabled:
                self._batched_profile_path_counts["semi_fast"] += 1
                num_groups = len(groups)
                savings = B - num_groups
                self._batched_profile_path_counts["semi_fast_groups"].append(num_groups)
                self._batched_profile_path_counts["semi_fast_savings"].append(savings)
                # Trim to max_samples
                if len(self._batched_profile_path_counts["semi_fast_groups"]) > self._batched_profile_max_samples:
                    self._batched_profile_path_counts["semi_fast_groups"] = self._batched_profile_path_counts[
                        "semi_fast_groups"
                    ][-self._batched_profile_max_samples :]
                    self._batched_profile_path_counts["semi_fast_savings"] = self._batched_profile_path_counts[
                        "semi_fast_savings"
                    ][-self._batched_profile_max_samples :]

            # Process each group with batched operations
            for (pop_out_len, spk_len, new_fifo_len), indices in groups.items():
                idx = torch.tensor(indices, device=device, dtype=torch.long)

                if pop_out_len > 0:
                    # Batched FIFO extraction
                    new_fifo_batch[idx, :new_fifo_len, :] = combined_fifo_batch[
                        idx, pop_out_len : pop_out_len + new_fifo_len, :
                    ]

                    # Batched spkcache extension
                    extended_spkcache_batch[idx, :spk_len, :] = spkcache_batch[idx, :spk_len, :]
                    extended_spkcache_batch[idx, spk_len : spk_len + pop_out_len, :] = combined_fifo_batch[
                        idx, :pop_out_len, :
                    ]

                    # Check for first-time (preds have -1.0 sentinel) - need to use TRT preds
                    # Vectorized: use torch.where instead of per-item loop
                    first_time_mask = spkcache_preds_batch[idx, 0, 0] < 0
                    extended_spkcache_preds_batch[idx, :spk_len, :] = torch.where(
                        first_time_mask.view(-1, 1, 1), preds[idx, :spk_len, :], spkcache_preds_batch[idx, :spk_len, :]
                    )

                    extended_spkcache_preds_batch[idx, spk_len : spk_len + pop_out_len, :] = combined_fifo_preds_batch[
                        idx, :pop_out_len, :
                    ]

                    # Process compression needs
                    for i in indices:
                        if needs_compression[i]:
                            compress_indices.append(i)
                            ext_len = spk_len + pop_out_len
                            compress_inputs_embs.append(extended_spkcache_batch[i, :ext_len, :])
                            compress_inputs_preds.append(extended_spkcache_preds_batch[i, :ext_len, :])
                        else:
                            new_spk_len = spk_len + pop_out_len
                            new_spkcache_batch[i, :new_spk_len, :] = extended_spkcache_batch[i, :new_spk_len, :]
                            new_spkcache_preds_batch[i, :new_spk_len, :] = extended_spkcache_preds_batch[
                                i, :new_spk_len, :
                            ]
                else:
                    # Batched no-pop path
                    new_fifo_batch[idx, :new_fifo_len, :] = combined_fifo_batch[idx, :new_fifo_len, :]
                    new_spkcache_batch[idx, :spk_len, :] = spkcache_batch[idx, :spk_len, :]
                    new_spkcache_preds_batch[idx, :spk_len, :] = spkcache_preds_batch[idx, :spk_len, :]

        if self._batched_profile_enabled:
            evt_after_fifo.record()

        # ========== PHASE 7: Async Batched Compression (Double-Buffer) ==========
        # Key insight: We CAN batch compression by padding to max length!
        # Padded frames have preds=0, so is_speech=False → scores=-inf → never selected by topk
        #
        # ASYNC: Compression runs on a separate stream, writes to pending buffers.
        # States are marked with has_pending_compression=True and synced before next use.
        compression_event = None
        compressed_indices_set = set(compress_indices)

        if compress_indices:
            # Find max length and pad all items to same length
            compress_lengths = [compress_inputs_embs[j].shape[0] for j in range(len(compress_indices))]
            max_compress_len = max(compress_lengths)
            num_compress = len(compress_indices)

            # Create compression stream lazily
            if self._compression_stream is None:
                self._compression_stream = torch.cuda.Stream()

            # Optimized: If all items have same length, use torch.stack (no loop)
            if len(set(compress_lengths)) == 1:
                # Uniform length - single batched stack operation
                compress_batch_embs = torch.stack(compress_inputs_embs, dim=0)
                compress_batch_preds = torch.stack(compress_inputs_preds, dim=0)
            else:
                # Variable lengths - use loop with zero-padding
                compress_batch_embs = torch.zeros(
                    (num_compress, max_compress_len, emb_dim), device=device, dtype=self.dtype
                )
                compress_batch_preds = torch.zeros(
                    (num_compress, max_compress_len, n_spk), device=device, dtype=self.dtype
                )
                for j in range(num_compress):
                    L = compress_lengths[j]
                    compress_batch_embs[j, :L, :] = compress_inputs_embs[j]
                    compress_batch_preds[j, :L, :] = compress_inputs_preds[j]

            # Wait for current stream work to complete before compression
            self._compression_stream.wait_stream(torch.cuda.current_stream())

            # Run compression asynchronously on compression stream
            with torch.cuda.stream(self._compression_stream):
                # Single batched compression call!
                compressed_embs, compressed_preds, _ = self._compress_spkcache(
                    emb_seq=compress_batch_embs, preds=compress_batch_preds, permute_spk=False
                )

                # Write results directly to each state's PENDING buffer
                # This happens on the compression stream (async to main inference)
                for j, idx in enumerate(compress_indices):
                    # Map to original batch index if we filtered by end_flags
                    orig_idx = updated_indices[idx] if updated_indices is not None else idx
                    state = batch_states[orig_idx]
                    # Write to pending buffers
                    state.spkcache_pending[0, : self.spkcache_len, :] = compressed_embs[j]
                    state.spkcache_preds_pending[0, : self.spkcache_len, :] = compressed_preds[j]
                    state.spkcache_len_pending = self.spkcache_len  # After compression, always full

                # Record event after all async work
                compression_event = torch.cuda.Event()
                compression_event.record()

            # Mark states as having pending compression (CPU operation, immediate)
            retained_tensors = (
                compress_batch_embs,
                compress_batch_preds,
                compressed_embs,
                compressed_preds,
            )
            # DEBUG: Force sync if flag is set (helps isolate async race conditions)
            if self._force_sync_compression:
                compression_event.synchronize()
                # Swap immediately since we've synced
                for idx in compress_indices:
                    orig_idx = updated_indices[idx] if updated_indices is not None else idx
                    state = batch_states[orig_idx]
                    state.spkcache, state.spkcache_pending = state.spkcache_pending, state.spkcache
                    state.spkcache_preds, state.spkcache_preds_pending = (
                        state.spkcache_preds_pending,
                        state.spkcache_preds,
                    )
                    state.spkcache_len_cached = state.spkcache_len_pending
                    # Clear pending flags since we already synced
                    state.has_pending_compression = False
                    state.compression_event = None
                    state.pending_compression_tensors = None
                # NOTE: Keep compressed_indices_set unchanged so Phase 8 skips these items
                # (their spkcache is already correct from the swap above)
            else:
                # Async path: mark states for later sync
                for idx in compress_indices:
                    orig_idx = updated_indices[idx] if updated_indices is not None else idx
                    batch_states[orig_idx].compression_event = compression_event
                    batch_states[orig_idx].has_pending_compression = True
                    batch_states[orig_idx].pending_compression_tensors = retained_tensors

        if self._batched_profile_enabled:
            # NOTE: For async compression, we don't sync here - compression continues in background
            evt_after_compress.record()

        # ========== PHASE 8: Write back to individual states ==========
        # Assign slices of batch tensors - these become views, no copy needed
        # NOTE: We only update the cached Python ints, not the GPU tensor lengths
        # The tensor lengths would cause BS*2 small CPU→GPU transfers
        #
        # ASYNC: For states with pending compression, skip spkcache assignment.
        # Their spkcache will be swapped in when sync_pending_compression is called.
        #
        # NOTE: When end_flags filtering is active, B is the filtered count and
        # updated_indices maps filtered index -> original batch index
        for i in range(B):
            # Map to original batch index if we filtered
            orig_idx = updated_indices[i] if updated_indices is not None else i

            batch_states[orig_idx].fifo = new_fifo_batch[i : i + 1]
            batch_states[orig_idx].fifo_len_cached = new_fifo_lens[i]

            if i not in compressed_indices_set:
                # Normal path: no compression, update spkcache directly
                batch_states[orig_idx].spkcache = new_spkcache_batch[i : i + 1]
                batch_states[orig_idx].spkcache_preds = new_spkcache_preds_batch[i : i + 1]
                batch_states[orig_idx].spkcache_len_cached = new_spk_lens[i]
            # else: state has pending compression, spkcache will be updated via swap after sync

        if self._batched_profile_enabled:
            evt_end.record()
            # Wait for all events to complete
            torch.cuda.synchronize()

            self._batched_profile_times["1_lengths"].append(evt_start.elapsed_time(evt_after_lengths))
            self._batched_profile_times["2_stack"].append(evt_after_lengths.elapsed_time(evt_after_stack))
            self._batched_profile_times["3_alloc"].append(evt_after_stack.elapsed_time(evt_after_alloc))
            self._batched_profile_times["4_gather"].append(evt_after_alloc.elapsed_time(evt_after_gather))
            self._batched_profile_times["5_fifo_loop"].append(evt_after_gather.elapsed_time(evt_after_fifo))
            self._batched_profile_times["6_compress"].append(evt_after_fifo.elapsed_time(evt_after_compress))
            self._batched_profile_times["7_writeback"].append(evt_after_compress.elapsed_time(evt_end))
            self._batched_profile_times["total"].append(evt_start.elapsed_time(evt_end))

            # Trim to max_samples
            for name in self._batched_profile_times:
                if len(self._batched_profile_times[name]) > self._batched_profile_max_samples:
                    self._batched_profile_times[name] = self._batched_profile_times[name][
                        -self._batched_profile_max_samples :
                    ]

            self._batched_profile_count += 1
            if self._batched_profile_count % self._batched_profile_interval == 0:
                print(
                    f"\n=== streaming_update_batched Profile (n={self._batched_profile_count}, B={B}) ===", flush=True
                )

                # Print path usage stats
                pc = self._batched_profile_path_counts
                total_paths = pc["fastest"] + pc["fast_full"] + pc["fast_varfifo"] + pc["semi_fast"]
                if total_paths > 0:
                    print(
                        f"  Path usage: fastest={pc['fastest']} ({100*pc['fastest']/total_paths:.1f}%), "
                        f"fast_full={pc['fast_full']} ({100*pc['fast_full']/total_paths:.1f}%), "
                        f"fast_varfifo={pc['fast_varfifo']} ({100*pc['fast_varfifo']/total_paths:.1f}%), "
                        f"semi_fast={pc['semi_fast']} ({100*pc['semi_fast']/total_paths:.1f}%)",
                        flush=True,
                    )
                    if pc["semi_fast_groups"]:
                        avg_groups = sum(pc["semi_fast_groups"]) / len(pc["semi_fast_groups"])
                        avg_savings = sum(pc["semi_fast_savings"]) / len(pc["semi_fast_savings"])
                        print(
                            f"  Semi-fast stats: avg_groups={avg_groups:.1f}, avg_savings={avg_savings:.1f} ops/call",
                            flush=True,
                        )

                for name, times in self._batched_profile_times.items():
                    if times:
                        recent = times[-self._batched_profile_max_samples :]
                        avg = sum(recent) / len(recent)
                        p50 = sorted(recent)[len(recent) // 2]
                        p99 = sorted(recent)[int(len(recent) * 0.99)]
                        print(f"  {name:15s}: avg={avg:.3f}ms, p50={p50:.3f}ms, p99={p99:.3f}ms", flush=True)
                print("=" * 60, flush=True)

        return batch_states, chunk_preds_out, updated_indices

    def streaming_update_async(self, streaming_state, chunk, chunk_lengths, preds, lc: int = 0, rc: int = 0):
        """
        Update the speaker cache and FIFO queue with the chunk of embeddings and speaker predictions.
        Asynchronous version, which means speaker cache, FIFO and chunk may have different lengths within a batch.
        Should be used for real streaming applications.

        Args:
            streaming_state (SortformerStreamingState): Previous streaming state including speaker cache and FIFO
            chunk (torch.Tensor): chunk of embeddings to be predicted
                Shape: (batch_size, lc+chunk_len+rc, emb_dim)
            chunk_lengths (torch.Tensor): Lengths of current chunk
                Shape: (batch_size,)
            preds (torch.Tensor): Speaker predictions of the [spkcache + fifo + chunk] embeddings
                Shape: (batch_size, spkcache_len + fifo_len + lc+chunk_len+rc, num_spks)
            lc and rc (int): The left & right offset of the chunk,
                only the chunk[:, lc:chunk_len+lc] is used for update of speaker cache and FIFO queue

        Returns:
            streaming_state (SortformerStreamingState): Current streaming state including speaker cache and FIFO
            chunk_preds (torch.Tensor): Speaker predictions of the chunk embeddings
                Shape: (batch_size, chunk_len, num_spks)
        """
        _, _, emb_dim = chunk.shape
        n_spk = preds.shape[2]

        max_spkcache_len, max_fifo_len, max_chunk_len = (
            streaming_state.spkcache.shape[1],
            streaming_state.fifo.shape[1],
            chunk.shape[1] - lc - rc,
        )

        if self.fifo_len == 0:
            max_pop_out_len = max_chunk_len
        elif self.spkcache_refresh_rate == 0:
            max_pop_out_len = self.fifo_len
        else:
            max_pop_out_len = min(max(self.spkcache_refresh_rate, max_chunk_len), self.fifo_len)

        fifo_preds = torch.zeros((1, max_fifo_len, n_spk), device=preds.device, dtype=self.dtype)
        chunk_preds = torch.zeros((1, max_chunk_len, n_spk), device=preds.device, dtype=self.dtype)
        chunk_lengths = (chunk_lengths - lc).clamp(min=0, max=max_chunk_len)
        updated_fifo = torch.zeros((1, max_fifo_len + max_chunk_len, emb_dim), device=preds.device, dtype=self.dtype)
        updated_fifo_preds = torch.zeros(
            (1, max_fifo_len + max_chunk_len, n_spk), device=preds.device, dtype=self.dtype
        )
        updated_spkcache = torch.zeros(
            (1, max_spkcache_len + max_pop_out_len, emb_dim), device=preds.device, dtype=self.dtype
        )
        updated_spkcache_preds = torch.full(
            (1, max_spkcache_len + max_pop_out_len, n_spk), 0.0, device=preds.device, dtype=self.dtype
        )

        batch_index = 0
        spkcache_len = streaming_state.spkcache_lengths[batch_index].item()
        fifo_len = streaming_state.fifo_lengths[batch_index].item()
        chunk_len = chunk_lengths[batch_index].item()
        fifo_preds[batch_index, :fifo_len, :] = preds[batch_index, spkcache_len : spkcache_len + fifo_len, :]
        chunk_preds[batch_index, :chunk_len, :] = preds[
            batch_index, spkcache_len + fifo_len + lc : spkcache_len + fifo_len + lc + chunk_len
        ]
        updated_spkcache[batch_index, :spkcache_len, :] = streaming_state.spkcache[batch_index, :spkcache_len, :]
        updated_spkcache_preds[batch_index, :spkcache_len, :] = streaming_state.spkcache_preds[
            batch_index, :spkcache_len, :
        ]
        updated_fifo[batch_index, :fifo_len, :] = streaming_state.fifo[batch_index, :fifo_len, :]
        updated_fifo_preds[batch_index, :fifo_len, :] = fifo_preds[batch_index, :fifo_len, :]

        # append chunk to fifo
        streaming_state.fifo_lengths[batch_index] += chunk_len
        updated_fifo[batch_index, fifo_len : fifo_len + chunk_len, :] = chunk[batch_index, lc : lc + chunk_len, :]
        updated_fifo_preds[batch_index, fifo_len : fifo_len + chunk_len, :] = chunk_preds[batch_index, :chunk_len, :]
        if fifo_len + chunk_len > max_fifo_len:
            # move pop_out_len first frames of FIFO queue to speaker cache
            pop_out_len = min(max_pop_out_len, fifo_len + chunk_len)
            streaming_state.spkcache_lengths[batch_index] += pop_out_len
            updated_spkcache[batch_index, spkcache_len : spkcache_len + pop_out_len, :] = updated_fifo[
                batch_index, :pop_out_len, :
            ]
            if updated_spkcache_preds[batch_index, 0, 0] >= 0:
                # speaker cache already compressed at least once
                updated_spkcache_preds[batch_index, spkcache_len : spkcache_len + pop_out_len, :] = updated_fifo_preds[
                    batch_index, :pop_out_len, :
                ]
            elif spkcache_len + pop_out_len > self.spkcache_len:
                # will compress speaker cache for the first time
                updated_spkcache_preds[batch_index, :spkcache_len, :] = preds[batch_index, :spkcache_len, :]
                updated_spkcache_preds[batch_index, spkcache_len : spkcache_len + pop_out_len, :] = updated_fifo_preds[
                    batch_index, :pop_out_len, :
                ]
            streaming_state.fifo_lengths[batch_index] -= pop_out_len
            new_fifo_len = streaming_state.fifo_lengths[batch_index].item()
            updated_fifo[batch_index, :new_fifo_len, :] = updated_fifo[
                batch_index, pop_out_len : pop_out_len + new_fifo_len, :
            ].clone()
            updated_fifo[batch_index, new_fifo_len:, :] = 0

        streaming_state.fifo = updated_fifo[:, :max_fifo_len, :]

        # update speaker cache
        streaming_state.spkcache = updated_spkcache[:, : self.spkcache_len :, :]
        streaming_state.spkcache_preds = updated_spkcache_preds[:, : self.spkcache_len :, :]

        need_compress = streaming_state.spkcache_lengths > self.spkcache_len
        idx = torch.where(need_compress)[0]

        if len(idx) > 0:
            streaming_state.spkcache[idx], streaming_state.spkcache_preds[idx], _ = self._compress_spkcache(
                emb_seq=updated_spkcache[idx], preds=updated_spkcache_preds[idx], permute_spk=False
            )
            streaming_state.spkcache_lengths[idx] = streaming_state.spkcache_lengths[idx].clamp(max=self.spkcache_len)

        # Update cached Python ints (critical for model.py which reads from these!)
        streaming_state.fifo_len_cached = int(streaming_state.fifo_lengths[batch_index].item())
        streaming_state.spkcache_len_cached = int(streaming_state.spkcache_lengths[batch_index].item())

        return streaming_state, chunk_preds[:, :chunk_len, :]

    def streaming_update(self, streaming_state, chunk, preds, lc: int = 0, rc: int = 0):
        """
        Update the speaker cache and FIFO queue with the chunk of embeddings and speaker predictions.
        Synchronous version, which means speaker cahce, FIFO queue and chunk have same lengths within a batch.
        Should be used for training and evaluation, not for real streaming applications.

        Args:
            streaming_state (SortformerStreamingState): previous streaming state including speaker cache and FIFO
            chunk (torch.Tensor): chunk of embeddings to be predicted
                Shape: (batch_size, lc+chunk_len+rc, emb_dim)
            preds (torch.Tensor): speaker predictions of the [spkcache + fifo + chunk] embeddings
                Shape: (batch_size, spkcache_len + fifo_len + lc+chunk_len+rc, num_spks)
            lc and rc (int): left & right offset of the chunk,
                only the chunk[:, lc:chunk_len+lc] is used for update of speaker cache and FIFO queue

        Returns:
            streaming_state (SortformerStreamingState): current streaming state including speaker cache and FIFO
            chunk_preds (torch.Tensor): speaker predictions of the chunk embeddings
                Shape: (batch_size, chunk_len, num_spks)
        """

        batch_size, _, emb_dim = chunk.shape

        spkcache_len, fifo_len, chunk_len = (
            streaming_state.spkcache.shape[1],
            streaming_state.fifo.shape[1],
            chunk.shape[1] - lc - rc,
        )
        if streaming_state.spk_perm is not None:
            inv_spk_perm = torch.stack(
                [torch.argsort(streaming_state.spk_perm[batch_index]) for batch_index in range(batch_size)]
            )
            preds = torch.stack(
                [preds[batch_index, :, inv_spk_perm[batch_index]] for batch_index in range(batch_size)]
            )

        streaming_state.fifo_preds = preds[:, spkcache_len : spkcache_len + fifo_len]
        chunk = chunk[:, lc : chunk_len + lc]
        chunk_preds = preds[:, spkcache_len + fifo_len + lc : spkcache_len + fifo_len + chunk_len + lc]

        # pop_out_len is the number of frames we will pop out from FIFO to update spkcache
        if self.fifo_len == 0:
            pop_out_len = chunk_len
        elif self.spkcache_refresh_rate == 0:
            pop_out_len = self.fifo_len
        else:
            pop_out_len = min(max(self.spkcache_refresh_rate, chunk_len), self.fifo_len)

        # append chunk to fifo
        streaming_state.fifo = torch.cat([streaming_state.fifo, chunk], dim=1)
        streaming_state.fifo_preds = torch.cat([streaming_state.fifo_preds, chunk_preds], dim=1)

        if fifo_len + chunk_len > self.fifo_len:
            # extract pop_out_len first frames from FIFO queue
            pop_out_len = min(pop_out_len, fifo_len + chunk_len)
            pop_out_embs = streaming_state.fifo[:, :pop_out_len]
            pop_out_preds = streaming_state.fifo_preds[:, :pop_out_len]
            streaming_state.fifo = streaming_state.fifo[:, pop_out_len:]
            streaming_state.fifo_preds = streaming_state.fifo_preds[:, pop_out_len:]

            # append pop_out_embs to spkcache
            streaming_state.spkcache = torch.cat([streaming_state.spkcache, pop_out_embs], dim=1)
            if streaming_state.spkcache_preds is not None:  # if speaker cache has been already updated at least once
                streaming_state.spkcache_preds = torch.cat([streaming_state.spkcache_preds, pop_out_preds], dim=1)
            if streaming_state.spkcache.shape[1] > self.spkcache_len:
                if streaming_state.spkcache_preds is None:  # if this is a first update of speaker cache
                    streaming_state.spkcache_preds = torch.cat([preds[:, :spkcache_len], pop_out_preds], dim=1)
                (
                    streaming_state.spkcache,
                    streaming_state.spkcache_preds,
                    streaming_state.spk_perm,
                ) = self._compress_spkcache(
                    emb_seq=streaming_state.spkcache, preds=streaming_state.spkcache_preds, permute_spk=self.training
                )

        if self.log:
            logging.info(
                f"spkcache: {streaming_state.spkcache.shape}, "
                f"chunk: {chunk.shape}, fifo: {streaming_state.fifo.shape}, "
                f"chunk_preds: {chunk_preds.shape}"
            )

        return streaming_state, chunk_preds

    def _boost_topk_scores(
        self, scores, n_boost_per_spk: int, scale_factor: float = 1.0, offset: float = 0.5
    ) -> torch.Tensor:
        """
        Increase `n_boost_per_spk` highest scores for each speaker.

        Args:
            scores (torch.Tensor): Tensor containing scores for each frame and speaker
                Shape: (batch_size, n_frames, n_spk)
            n_boost_per_spk (int): Number of frames to boost per speaker
            scale_factor (float): Scaling factor for boosting scores. Defaults to 1.0.
            offset (float): Offset for score adjustment. Defaults to 0.5.

        Returns:
            scores (torch.Tensor): Tensor containing scores for each frame and speaker after boosting.
                Shape: (batch_size, n_frames, n_spk)
        """
        batch_size, _, n_spk = scores.shape
        _, topk_indices = torch.topk(scores, n_boost_per_spk, dim=1, largest=True, sorted=False)
        batch_indices = torch.arange(batch_size).unsqueeze(1).unsqueeze(2)  # Shape: (batch_size, 1, 1)
        speaker_indices = torch.arange(n_spk).unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, n_spk)
        # Boost scores corresponding to topk_indices; but scores for disabled frames will remain '-inf'
        scores[batch_indices, topk_indices, speaker_indices] -= scale_factor * math.log(offset)
        return scores

    def _get_silence_profile(self, emb_seq, preds):
        """
        Get mean silence embedding from emb_seq sequence.
        Embeddings are considered as silence if sum of corresponding preds is lower than self.sil_threshold.

        Args:
            emb_seq (torch.Tensor): Tensor containing sequence of embeddings
                Shape: (batch_size, n_frames, emb_dim)
            preds (torch.Tensor): Tensor containing speaker activity probabilities
                Shape: (batch_size, n_frames, n_spk)

        Returns:
            mean_sil_emb (torch.Tensor): Mean silence embedding tensor
                Shape: (batch_size, emb_dim)
        """
        is_sil = preds.sum(dim=2) < self.sil_threshold
        is_sil = is_sil.unsqueeze(-1)
        emb_seq_sil = torch.where(is_sil, emb_seq, torch.zeros_like(emb_seq))  # (batch_size, n_frames, emb_dim)
        emb_seq_sil_sum = emb_seq_sil.sum(dim=1)  # (batch_size, emb_dim)
        sil_count = is_sil.sum(dim=1).clamp(min=1)  # (batch_size)
        mean_sil_emb = emb_seq_sil_sum / sil_count  # (batch_size, emb_dim)
        return mean_sil_emb

    def _get_log_pred_scores(self, preds):
        """
        Get per-frame scores for speakers based on their activity probabilities.
        Scores are log-based and designed to be high for confident prediction of non-overlapped speech.

        Args:
            preds (torch.Tensor): Tensor containing speaker activity probabilities
                Shape: (batch_size, n_frames, n_spk)

        Returns:
            scores (torch.Tensor): Tensor containing speaker scores
                Shape: (batch_size, n_frames, n_spk)
        """
        log_probs = torch.log(torch.clamp(preds, min=self.pred_score_threshold))
        log_1_probs = torch.log(torch.clamp(1.0 - preds, min=self.pred_score_threshold))
        log_1_probs_sum = log_1_probs.sum(dim=2).unsqueeze(-1).expand(-1, -1, self.n_spk)
        scores = log_probs - log_1_probs + log_1_probs_sum - math.log(0.5)
        return scores

    def _get_topk_indices(self, scores):
        """
        Get indices corresponding to spkcache_len highest scores, and binary mask for frames in topk to be disabled.
        Disabled frames correspond to either '-inf' score or spkcache_sil_frames_per_spk frames of extra silence
        Mean silence embedding will be used for these frames.

        Args:
            scores (torch.Tensor): Tensor containing speaker scores, including for extra silence frames
                Shape: (batch_size, n_frames, n_spk)

        Returns:
            topk_indices_sorted (torch.Tensor): Tensor containing frame indices of spkcache_len highest scores
                Shape: (batch_size, spkcache_len)
            is_disabled (torch.Tensor): Tensor containing binary mask for frames in topk to be disabled
                Shape: (batch_size, spkcache_len)
        """
        batch_size, n_frames, _ = scores.shape
        n_frames_no_sil = n_frames - self.spkcache_sil_frames_per_spk
        # Concatenate scores for all speakers and get spkcache_len frames with highest scores.
        # Replace topk_indices corresponding to '-inf' score with a placeholder index self.max_index.
        scores_flatten = scores.permute(0, 2, 1).reshape(batch_size, -1)
        topk_values, topk_indices = torch.topk(scores_flatten, self.spkcache_len, dim=1, sorted=False)
        valid_topk_mask = topk_values != float('-inf')
        topk_indices = torch.where(
            valid_topk_mask, topk_indices, torch.tensor(self.max_index, device=topk_indices.device)
        )
        # Sort topk_indices to preserve the original order of the frames.
        # Get correct indices corresponding to the original frames
        topk_indices_sorted, _ = torch.sort(topk_indices, dim=1)  # Shape: (batch_size, spkcache_len)
        is_disabled = topk_indices_sorted == self.max_index
        topk_indices_sorted = torch.remainder(topk_indices_sorted, n_frames)
        is_disabled += topk_indices_sorted >= n_frames_no_sil
        topk_indices_sorted[is_disabled] = 0  # Set a placeholder index to make gather work
        return topk_indices_sorted, is_disabled

    def _gather_spkcache_and_preds(self, emb_seq, preds, topk_indices, is_disabled):
        """
        Gather embeddings from emb_seq and speaker activities from preds corresponding to topk_indices.
        For disabled frames, use mean silence embedding and zero probability instead.

        Args:
            emb_seq (torch.Tensor): Tensor containing sequence of embeddings.
                Shape: (batch_size, n_frames, emb_dim)
            preds (torch.Tensor): Tensor containing speaker activity probabilities
                Shape: (batch_size, n_frames, n_spk)
            topk_indices (torch.Tensor): Tensor containing indices of frames to gather
                Shape: (batch_size, spkcache_len)
            is_disabled (torch.Tensor): Tensor containing binary mask for disabled frames
                Shape: (batch_size, spkcache_len)

        Returns:
            emb_seq_gathered (torch.Tensor): Tensor containing gathered embeddings.
                Shape: (batch_size, spkcache_len, emb_dim)
            preds_gathered (torch.Tensor): Tensor containing gathered speaker activities.
                Shape: (batch_size, spkcache_len, n_spk)
        """
        # To use `torch.gather`, expand `topk_indices` along the last dimension to match `emb_dim`.
        # Gather the speaker cache embeddings, including the placeholder embeddings for silence frames.
        # Finally, replace the placeholder embeddings with actual mean silence embedding.
        emb_dim, n_spk = emb_seq.shape[2], preds.shape[2]
        indices_expanded_emb = topk_indices.unsqueeze(-1).expand(-1, -1, emb_dim)
        emb_seq_gathered = torch.gather(emb_seq, 1, indices_expanded_emb)  # (batch_size, spkcache_len, emb_dim)
        mean_sil_emb = self._get_silence_profile(emb_seq, preds)  # Compute mean silence embedding
        mean_sil_emb_expanded = mean_sil_emb.unsqueeze(1).expand(-1, self.spkcache_len, -1)
        emb_seq_gathered = torch.where(is_disabled.unsqueeze(-1), mean_sil_emb_expanded, emb_seq_gathered)

        # To use `torch.gather`, expand `topk_indices` along the last dimension to match `n_spk`.
        # Gather speaker cache predictions `preds`, including the placeholder `preds` for silence frames.
        # Finally, replace the placeholder `preds` with zeros.
        indices_expanded_spk = topk_indices.unsqueeze(-1).expand(-1, -1, n_spk)
        preds_gathered = torch.gather(preds, 1, indices_expanded_spk)  # (batch_size, spkcache_len, n_spk)
        preds_gathered = torch.where(is_disabled.unsqueeze(-1), torch.tensor(0.0, device=preds.device), preds_gathered)
        return emb_seq_gathered, preds_gathered

    def _get_max_perm_index(self, scores):
        """
        Get number of first speakers having at least one positive score.
        These speakers will be randomly permuted during _compress_spkcache (training only).

        Args:
            scores (torch.Tensor): Tensor containing speaker scores
                Shape: (batch_size, n_frames, n_spk)

        Returns:
            max_perm_index (torch.Tensor): Tensor with number of first speakers to permute
                Shape: (batch_size)
        """

        batch_size, _, n_spk = scores.shape
        is_pos = scores > 0  # positive score usually means that only current speaker is speaking
        zero_indices = torch.where(is_pos.sum(dim=1) == 0)
        max_perm_index = torch.full((batch_size,), n_spk, dtype=torch.long, device=scores.device)
        max_perm_index.scatter_reduce_(0, zero_indices[0], zero_indices[1], reduce="amin", include_self=False)
        return max_perm_index

    def _disable_low_scores(self, preds, scores, min_pos_scores_per_spk: int):
        """
        Sets scores for non-speech to '-inf'.
        Also sets non-positive scores to '-inf', if there are at least min_pos_scores_per_spk positive scores.

        Args:
            preds (torch.Tensor): Tensor containing speaker activity probabilities
                Shape: (batch_size, n_frames, n_spk)
            scores (torch.Tensor): Tensor containing speaker importance scores
                Shape: (batch_size, n_frames, n_spk)
            min_pos_scores_per_spk (int): if number of positive scores for a speaker is greater than this,
                then all non-positive scores for this speaker will be disabled, i.e. set to '-inf'.

        Returns:
            scores (torch.Tensor): Tensor containing speaker scores.
                Shape: (batch_size, n_frames, n_spk)
        """
        # Replace scores for non-speech with '-inf'.
        is_speech = preds > 0.5
        scores = torch.where(is_speech, scores, torch.tensor(float('-inf'), device=scores.device))

        # Replace non-positive scores (usually overlapped speech) with '-inf'
        # This will be applied only if a speaker has at least min_pos_scores_per_spk positive-scored frames
        is_pos = scores > 0  # positive score usually means that only current speaker is speaking
        is_nonpos_replace = (~is_pos) * is_speech * (is_pos.sum(dim=1).unsqueeze(1) >= min_pos_scores_per_spk)
        scores = torch.where(is_nonpos_replace, torch.tensor(float('-inf'), device=scores.device), scores)
        return scores

    def _permute_speakers(self, scores, max_perm_index):
        """
        Create a random permutation of scores max_perm_index first speakers.

        Args:
            scores (torch.Tensor): Tensor containing speaker scores
                Shape: (batch_size, n_frames, n_spk)
            max_perm_index (torch.Tensor): Tensor with number of first speakers to permute
                Shape: (batch_size)

        Returns:
            scores (torch.Tensor): Tensor with permuted scores.
                Shape: (batch_size, n_frames, n_spk)
            spk_perm (torch.Tensor): Tensor containing speaker permutation applied to scores
                Shape: (batch_size, n_spk)
        """
        spk_perm_list, scores_list = [], []
        batch_size, _, n_spk = scores.shape
        for batch_index in range(batch_size):
            rand_perm_inds = torch.randperm(max_perm_index[batch_index].item())
            linear_inds = torch.arange(max_perm_index[batch_index].item(), n_spk)
            permutation = torch.cat([rand_perm_inds, linear_inds])
            spk_perm_list.append(permutation)
            scores_list.append(scores[batch_index, :, permutation])
        spk_perm = torch.stack(spk_perm_list).to(scores.device)
        scores = torch.stack(scores_list).to(scores.device)
        return scores, spk_perm

    def _compress_spkcache(self, emb_seq, preds, permute_spk: bool = False):
        """
        Compress speaker cache for streaming inference.
        Keep spkcache_len most important frames out of input n_frames, based on preds.

        Args:
            emb_seq (torch.Tensor): Tensor containing n_frames > spkcache_len embeddings
                Shape: (batch_size, n_frames, emb_dim)
            preds (torch.Tensor): Tensor containing n_frames > spkcache_len speaker activity probabilities
                Shape: (batch_size, n_frames, n_spk)
            permute_spk (bool): If true, will generate a random permutation of existing speakers

        Returns:
            spkcache (torch.Tensor): Tensor containing spkcache_len most important embeddings from emb_seq.
                Embeddings are ordered by speakers. Within each speaker, original order of frames is kept.
                Shape: (batch_size, spkcache_len, emb_dim)
            spkcache_preds (torch.Tensor): predictions corresponding to speaker cache
                Shape: (batch_size, spkcache_len, n_spk)
            spk_perm (torch.Tensor): random speaker permutation tensor if permute_spk=True, otherwise None
                Shape: (batch_size, n_spk)
        """
        batch_size, n_frames, n_spk = preds.shape
        spkcache_len_per_spk = self.spkcache_len // n_spk - self.spkcache_sil_frames_per_spk
        strong_boost_per_spk = math.floor(spkcache_len_per_spk * self.strong_boost_rate)
        weak_boost_per_spk = math.floor(spkcache_len_per_spk * self.weak_boost_rate)
        min_pos_scores_per_spk = math.floor(spkcache_len_per_spk * self.min_pos_scores_rate)

        scores = self._get_log_pred_scores(preds)
        scores = self._disable_low_scores(preds, scores, min_pos_scores_per_spk)

        if permute_spk:  # Generate a random permutation of speakers
            max_perm_index = self._get_max_perm_index(scores)
            scores, spk_perm = self._permute_speakers(scores, max_perm_index)
        else:
            spk_perm = None

        if self.scores_boost_latest > 0:  # Boost newly added frames
            scores[:, self.spkcache_len :, :] += self.scores_boost_latest

        # if self.training:
        #     if self.scores_add_rnd > 0:  # Add random noise to scores
        #         scores += torch.rand(batch_size, n_frames, n_spk, device=scores.device) * self.scores_add_rnd

        # Strong boosting to ensure each speaker has at least K frames in speaker cache
        scores = self._boost_topk_scores(scores, strong_boost_per_spk, scale_factor=2)
        # Weak boosting to prevent dominance of one speaker in speaker cache
        scores = self._boost_topk_scores(scores, weak_boost_per_spk, scale_factor=1)

        if self.spkcache_sil_frames_per_spk > 0:  # Add number of silence frames in the end of each block
            pad = torch.full((batch_size, self.spkcache_sil_frames_per_spk, n_spk), float('inf'), device=scores.device)
            scores = torch.cat([scores, pad], dim=1)  # (batch_size, n_frames + spkcache_sil_frames_per_spk, n_spk)

        topk_indices, is_disabled = self._get_topk_indices(scores)
        spkcache, spkcache_preds = self._gather_spkcache_and_preds(emb_seq, preds, topk_indices, is_disabled)
        return spkcache, spkcache_preds, spk_perm
