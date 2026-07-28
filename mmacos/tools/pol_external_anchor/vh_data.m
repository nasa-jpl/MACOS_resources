function d = vh_data()
%VH_DATA  Published inputs for the protected-aluminum external anchor.
%
%   Single source of truth for every number taken from the publication, so
%   the harness and the gate cannot drift apart.  NOTHING here is fitted,
%   tuned, or measured by us -- each field records where it came from.
%
%   SOURCE
%     G. van Harten, F. Snik & C. U. Keller,
%     "Polarization properties of real aluminum mirrors
%      I. Influence of the aluminum oxide layer",
%     Publications of the Astronomical Society of the Pacific 121, 377-383
%     (2009).  doi:10.1086/599043.  Preprint arXiv:0903.2740v1.
%     Captured 2026-07-28 from https://arxiv.org/pdf/0903.2740 (v1).
%
%   WHY THIS PAPER.  It is the rare combination the anchor needs: a
%   protected/oxidized metal mirror whose FULL Mueller matrix was measured
%   by ellipsometry over a swept incidence angle at several wavelengths,
%   AND whose model inputs (both indices, the film thickness, the film
%   model itself) are stated numerically in the paper.  That lets the
%   engine be driven with THEIR inputs and compared against THEIR curves,
%   which isolates the thin-film machinery from index-table disagreements.
%   Index tables for aluminum genuinely disagree with one another -- the
%   paper says so itself ("the values of k are widely varying throughout
%   the literature") and fits k rather than adopting a table.  A
%   disagreement traceable to index tables is NOT a machinery error, and
%   this construction cannot confuse the two.
%
%   MEASUREMENT (their Sec. 2), for the record:
%     * complete Mueller-matrix ellipsometer, liquid-crystal variable
%       retarders, eigenvalue calibration method (Compain et al. 1999)
%     * 220 +- 10 nm Al evaporated on a 5x5 cm glass substrate
%     * 14 incidence angles spanning 6-70 deg
%     * lambda = 500, 550, 600, 650 nm, 10 nm bandpass
%     * polarimetric sensitivity 2e-4; ABSOLUTE ACCURACY ~1% of element
%       [1,1], i.e. +-0.01 per normalized Mueller matrix element.  That
%       +-0.01 is the number the gate tolerance is stated against.

    % ---- Table 2 ---------------------------------------------------------
    % Literature columns:
    %   nf   : amorphous Al2O3 index, +-0.01, Eriksson et al. (1981)
    %   n_al : real part of the Al index, +-0.01, Lide (2008),
    %          linearly interpolated to these wavelengths
    % Determined column (their "Fit 1" = fit to k and d with literature
    % n_al, i.e. the fit that INCLUDES the oxide layer -- the model this
    % anchor reproduces):
    %   k_al : imaginary part of the Al index, with its 1-sigma fit error
    d.lambda_nm = [500,   550,   600,   650  ];
    d.nf        = [1.61,  1.61,  1.60,  1.60 ];
    d.n_al      = [0.769, 0.958, 1.200, 1.470];
    d.k_al      = [5.88,  6.30,  6.85,  7.33 ];
    d.k_al_err  = [0.02,  0.03,  0.03,  0.03 ];

    d.n_err     = 0.01;     % on nf and n_al alike (both "+-0.01" in Table 2)

    % ---- fitted oxide thickness (their abstract + Sec. 4) ----------------
    % "remains stable at a value of 4.12 +- 0.08 nm on the long term"
    d.d_oxide_nm     = 4.12;
    d.d_oxide_err_nm = 0.08;

    % ---- the aluminum film as evaporated (their Sec. 2) ------------------
    % The paper MODELS the metal as semi-infinite ("bulk aluminum"); the
    % actual sample was a 220 nm film.  At k ~ 6-7 the field is attenuated
    % by e^-1 in lambda/(4 pi k) ~ 7 nm, so 220 nm is ~30 absorption depths
    % and the two are numerically identical (the harness verifies this
    % rather than asserting it).  We drive the engine with the REAL film,
    % because coat_set's substrate slot is the element's own IndRef and
    % there is no API to set that -- see README.
    d.d_al_nm     = 220.0;
    d.d_al_err_nm = 10.0;

    % ---- incidence-angle range actually measured -------------------------
    d.aoi_min_deg = 6;
    d.aoi_max_deg = 70;
    d.n_aoi       = 14;

    % ---- their stated absolute accuracy ----------------------------------
    % "All measurements presented here exhibit an absolute accuracy of ~1%
    %  of element [1,1], i.e. +-0.01 per normalized Mueller matrix element"
    d.mueller_accuracy = 0.01;

    % ---- provenance ------------------------------------------------------
    d.source    = ['van Harten, Snik & Keller, PASP 121, 377 (2009); ' ...
                   'arXiv:0903.2740v1, Table 2 + Secs. 2-4'];
    d.doi       = '10.1086/599043';
    d.captured  = '2026-07-28';
end
