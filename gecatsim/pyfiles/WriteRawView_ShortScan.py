# Copyright 2024, GE Precision HealthCare. All rights reserved. See https://github.com/xcist/main/tree/master/license

'''

'''
import numpy as np

def WriteRawView_ShortScan(cfg, viewId):
    
    # --------------------------
    # 1. Compute the gantry angle
    # --------------------------
    startAngle = cfg.protocol.startAngle * np.pi/180
    viewsPerRot = cfg.protocol.viewsPerRotation
    rotDir = cfg.protocol.rotationDirection

    beta = startAngle + (viewId / viewsPerRot) * rotDir * 2*np.pi
    beta_deg = np.degrees(beta % (2*np.pi))

    # --------------------------
    # 2. Define valid angular range
    # --------------------------
    allowed_start = 0
    allowed_end   = cfg.protocol.coverage  # short-scan width

    # Outside this range → partial-detector masking required
    if beta_deg < allowed_start or beta_deg > allowed_end:
        # cfg.thisView shape = [col, row] or [col, row, energy]
        view = cfg.thisView.copy()

        # -------------------------------
        # 3. Compute triangular fade-out:
        #    only some detector columns
        #    are outside the coverage
        # -------------------------------
        cols = cfg.scanner.detectorColCount
        
        # Normalize angle difference
        if beta_deg < allowed_start:
            delta = allowed_start - beta_deg
        else:
            delta = beta_deg - allowed_end

        # max fade-out width (tunable)
        max_missing_det = int(cols * (delta / (360 - (allowed_end - allowed_start))))

        # clamp
        max_missing_det = max(0, min(cols, max_missing_det))

        # ------------------------------
        # 4. Zero the *edge* detectors
        # ------------------------------
        # Example: zero detectors from right edge inward
        if max_missing_det > 0:
            view[-max_missing_det:, :] = 0.0

        cfg.thisView = view

    # ----------------------------------
    # 5. Save to raw (same as XCIST)
    # ----------------------------------
    scanTypeInd = [cfg.sim.isAirScan, cfg.sim.isOffsetScan, cfg.sim.isPhantomScan].index(1)
    extName = ['.air', '.offset', '.scan'][scanTypeInd]
    fname = cfg.resultsName + extName

    # reshape for saving
    if cfg.thisView.ndim == 1:
        dims = [cfg.scanner.detectorColCount, cfg.scanner.detectorRowCount]
        thisView = cfg.thisView.reshape(dims).T.ravel()
    else:
        dims = [cfg.scanner.detectorColCount, cfg.scanner.detectorRowCount, cfg.thisView.shape[1]]
        thisView = cfg.thisView.reshape(dims).transpose((1,0,2)).ravel()

    # write
    accessMode = 'wb' if viewId == cfg.sim.startViewId else 'ab'
    with open(fname, accessMode) as f:
        f.write(thisView)

