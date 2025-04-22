<!-- /vibe-player/TODO.md -->
# Vibe Player - TODO & Future Ideas

This file tracks potential improvements, features, and known issues requiring further investigation for the Vibe Player project.

## Bugs / Issues

*   **[INVESTIGATE] Formant Shift:** The formant shift feature provided by Rubberband WASM is currently non-functional (no audible effect). Requires deeper investigation. *(Deferred)*
*   **[RESOLVED - Post-MVP Fix] Pause Position Bug:** Play position jumped back on pause. Fixed by correcting the order of operations in `app.js::handlePlayPause`.
*   **[RESOLVED - Post-MVP Fix] Delay Change Bug:** Changing delay offset while playing didn't affect audio. Fixed by triggering a seek in `app.js::handleDelayChange`.

## Immediate Implementation (Multi-Track Feature Set & Refactor)

**Goal:** Implement core multi-track features (Volume, Mute, Solo, Swap, Multi-VAD) on a refactored architecture using track data indirection for robustness and extensibility.

**Order of Operations:**

1.  **Phase R1: Architectural Refactor - Track Indirection (NEXT)**
    *   **Goal:** Modify the core application structure to manage track data independently from UI channel assignment (Left/Right).
    *   **Tasks:**
        *   [ ] **`app.js`:** Refactor state to use `tracksData` array and `left/rightChannelTrackIndex` pointers. Adapt helper functions.
        *   [ ] **`audioEngine.js`:** Modify to accept numeric `trackId`s (0, 1, ...) instead of string IDs (`'track_left'`). Update internal map, worklet communication, and public methods accordingly.
        *   [ ] **`app.js`:** Update event handlers and logic to map UI sides ('left'/'right') to track indices when accessing `tracksData` or calling `audioEngine`.
        *   [ ] **`uiManager.js`:** Adapt UI update functions as needed. Implement necessary logic for refreshing UI elements based on current L/R track index assignments (e.g., `refreshTrackUI(trackSide, trackIndex)` helper).
        *   [ ] **`app.js`:** Implement the simplified `handleSwapTracks` using index pointer swapping and UI refresh.
    *   **(Testing Step)** Verify loading, basic playback, seeking, linked speed/pitch, and swap work correctly after refactor.

2.  **Phase M1: Mute & Solo Implementation (After Refactor)**
    *   **Goal:** Implement working Mute and Solo buttons for individual tracks.
    *   **Tasks:**
        *   [ ] **`app.js`:** Implement `handleMuteToggle`, `handleSoloToggle` to update `isMuted`/`isSoloed` flags in `tracksData`.
        *   [ ] **`app.js`:** Implement `applyMuteSoloState` helper function to calculate effective mute state based on all track flags and call `audioEngine.setMute`.
        *   [ ] **`uiManager.js`:** Implement `setMuteButtonState`/`setSoloButtonState` for visual feedback. Ensure listeners call `app.js` handlers.
    *   **(Testing Step)**

3.  **Phase M2: Right Track VAD (After Mute/Solo)**
    *   **Goal:** Enable VAD processing and waveform highlighting for the Right track.
    *   **Tasks:**
        *   [ ] **`app.js`:** Modify `runVadInBackground` to be track-agnostic (accept `trackIndex`). Call it for the right track index upon its `audioLoaded`.
        *   [ ] **`app.js`:** Ensure VAD results are stored in the correct `tracksData[index].vad`. Update the correct visualizer (`vizRefs.waveform`).
        *   [ ] **`uiManager.js`:** Update VAD Tuning section title to clarify sliders only affect Left Track VAD recalculation (Decision: Option A - Simpler). Ensure VAD progress/status text indicates the correct track ('Left' or 'Right').
    *   **(Testing Step)**

4.  **Phase M3: Individual Volume & Unlinked Pitch (Final MVP Features)**
    *   **Goal:** Ensure individual track volume sliders work correctly and verify unlinked pitch.
    *   **Tasks:**
        *   [ ] **`app.js`:** Ensure `handleVolumeChange` correctly maps the UI side (`_left`/`_right`) to the appropriate `trackIndex` and updates `tracksData[trackIndex].parameters.volume`. Ensure it calls `audioEngine.setVolume(trackIndex, newVolume)`.
        *   [ ] **Testing:** Thoroughly test unlinked pitch mode toggle and control.
        *   [ ] **Testing:** Thoroughly test individual volume sliders.
    *   **(Testing Step - Final MVP Features)**

## Deferred Features / Enhancements (Post-MVP)

*   **UI / UX:**
    *   **Individual Speed:** Add UI and logic for unlinked Speed controls (requires significant timing logic changes).
    *   **XP.css Theme:** Upgrade styling from 98.css to XP.css (or similar).
    *   **Windows 98 Sounds:** Add nostalgic UI sound effects on interactions.
    *   **Parameter Smoothing:** Investigate smoother transitions for Speed/Pitch changes.
    *   **Error Handling UI:** Display user-friendly error messages in the UI overlay/section.
    *   **Preset Management:** Allow saving/loading sets of Speed/Pitch/Gain/Volume/Offset/VAD settings.
    *   **Keyboard Bindings:** Make keybinds configurable (UI + LocalStorage). Add Reset button. Add more bindings (e.g., speed up/down, reset controls, Mute/Solo L/R). Add 'Back to Start' button.
    *   **Drift Display:** Implement actual calculation and display of ms drift between tracks (reliant on worklet time reports). *(Partially done, needs verification)*
*   **Visualizations:**
    *   **Graphical Offset:** Implement shifting/scaling of waveform/spectrogram content based on offset.
    *   **Toggle Viz Type:** Allow toggling track display between Waveform and Spectrogram.
    *   **VAD Probability Graph:** Add graph showing VAD probabilities and thresholds (likely Left only initially).
    *   **VAD Threshold Controls for Right Track:** Add separate UI controls if desired (Option B).
*   **Audio / Sync:**
    *   **Active Drift Correction:** Implement periodic re-syncing via seek *if testing shows significant drift*. Add manual "Resync" button first.
*   **Architecture / Code Health:**
    *   **VAD Worker:** Migrate VAD processing to a Web Worker.
    *   **Visualizer Computation Worker(s):** Offload waveform/spectrogram calculation.
    *   **State Management Module (`audioPlayerState.js`):** Refactor state logic out of `app.js`.
    *   **Review `app.js` / `audioEngine.js` Complexity:** Monitor and refactor as features grow.
    *   **Automated Testing:** Introduce unit/integration tests.
    *   **Spectrogram Cache:** Verify robustness with multiple instances.
    *   **Track Deletion/Reordering:** Consider support beyond just L/R swap.

<!-- /vibe-player/TODO.md -->