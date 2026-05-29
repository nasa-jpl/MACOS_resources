#include "fintrf.h"
C=======================================================================
C
C     mmacos_mex.F -- MATLAB mex bridge to MACOS via macos_api_mod
C
C     Command-dispatched mex.  MATLAB callers write:
C
C       mmacos('init', 128)              % allocate macos at model size
C       n = mmacos('load_rx', 'foo.in')  % load prescription
C       mmacos('modified_rx')            % equivalent to MODIFY command
C       opd = mmacos('opd', 11)          % OPDMat at element 11
C       cf  = mmacos('complex_field', 9) % WFElt at element 9 (complex)
C       dx  = mmacos('dx_at', 9)         % per-element dx in SI metres
C       mmacos('perturb_elt', [2], [Rx Ry Rz Tx Ty Tz]', true)
C
C     The mex shares all state with pymacos's macos_api_mod (which is
C     the language-neutral SMACOS-call wrapper produced by §5.1).  This
C     means a single libsmacos.a backs both Python and MATLAB.
C
C     MVP command set (others land as needed):
C       init                  Initialize/resize macos
C       load_rx               'OLD' a prescription
C       save_rx               'SAVE' to a prescription
C       modified_rx           'MODIFY' (reset ray-trace state)
C       n_elt                 # elements in current Rx
C       trace_rays            Full TRACE -- needed before opd/intensity/etc
C       opd                   OPDMat from last trace  (N x N real)
C       intensity             |WFElt|^2 at element  (after INT)
C       complex_field         WFElt at element  (complex)
C       dx_at                 per-element diffraction-grid dx (metres)
C       base_unit_to_metres   CBM conversion factor
C       apodize               in-place multiply WFElt by real mask
C       perturb_elt           Rigid-body PERTURB of one or more elements
C
C=======================================================================
        subroutine mexfunction(nlhs, plhs, nrhs, prhs)

        implicit none

        mwPointer plhs(*), prhs(*)
        integer nlhs, nrhs

        ! MATLAB API function decls
        mwPointer mxGetPr, mxCreateDoubleMatrix, mxCreateDoubleScalar
        mwPointer mxGetM, mxGetN
        integer*4 mxIsChar, mxIsDouble, mxIsLogical
        integer*4 mxGetString
        mwSize    mxGetNumberOfElements

        ! Local
        character(len=64) :: cmd
        integer*4         :: stat
        mwSize            :: cmdlen
        logical           :: handled_gen

        IF (nrhs .lt. 1) THEN
          CALL mexErrMsgTxt('mmacos requires at least one argument: the command string.')
        END IF
        IF (mxIsChar(prhs(1)) .ne. 1) THEN
          CALL mexErrMsgTxt('mmacos: first argument must be a command string')
        END IF

        cmdlen = mxGetM(prhs(1)) * mxGetN(prhs(1))
        IF (cmdlen .gt. len(cmd)) THEN
          CALL mexErrMsgTxt('mmacos: command string too long')
        END IF
        cmd = ' '
        stat = mxGetString(prhs(1), cmd, int(len(cmd), kind=4))
        IF (stat .ne. 0) THEN
          CALL mexErrMsgTxt('mmacos: failed to read command string')
        END IF

        SELECT CASE (trim(cmd))
          CASE ('init')
            CALL do_init(nlhs, plhs, nrhs, prhs)
          CASE ('load_rx')
            CALL do_load_rx(nlhs, plhs, nrhs, prhs)
          CASE ('save_rx')
            CALL do_save_rx(nlhs, plhs, nrhs, prhs)
          CASE ('modified_rx')
            CALL do_modified_rx(nlhs, plhs, nrhs, prhs)
          CASE ('n_elt')
            CALL do_n_elt(nlhs, plhs, nrhs, prhs)
          CASE ('opd')
            CALL do_opd(nlhs, plhs, nrhs, prhs)
          CASE ('intensity')
            CALL do_intensity(nlhs, plhs, nrhs, prhs)
          CASE ('complex_field')
            CALL do_complex_field(nlhs, plhs, nrhs, prhs)
          CASE ('dx_at')
            CALL do_dx_at(nlhs, plhs, nrhs, prhs)
          CASE ('base_unit_to_metres')
            CALL do_cbm(nlhs, plhs, nrhs, prhs)
          CASE ('apodize')
            CALL do_apodize(nlhs, plhs, nrhs, prhs)
          CASE ('perturb_elt')
            CALL do_perturb_elt(nlhs, plhs, nrhs, prhs)
          CASE ('trace_rays')
            CALL do_trace_rays(nlhs, plhs, nrhs, prhs)
          CASE DEFAULT
            ! Fall through to auto-generated dispatcher.  Covers every
            ! macos_api_mod routine NOT special-cased above; see
            ! mmacos_gen.F (produced by gen_mex_wrappers.py).
            CALL gen_dispatch(cmd, nlhs, plhs, nrhs, prhs, handled_gen)
            IF (.not. handled_gen) THEN
              ! Assemble error string in a buffer because cpp eats `//`
              ! at end of preprocessed line in fixed-form sources.
              BLOCK
                CHARACTER(len=128) :: errmsg
                errmsg = 'mmacos: unknown command "' // trim(cmd) // '"'
                CALL mexErrMsgTxt(trim(errmsg))
              END BLOCK
            END IF
        END SELECT

        return
        end subroutine mexfunction

C=======================================================================
C     init                                                              =
C=======================================================================
        subroutine do_init(nlhs, plhs, nrhs, prhs)
        use macos_api_mod, only: init
        implicit none
        mwPointer plhs(*), prhs(*)
        integer nlhs, nrhs

        mwPointer mxGetPr
        mwPointer pr
        real(8)   buf(1)
        integer   model_size
        logical   ok

        IF (nrhs .ne. 2) CALL mexErrMsgTxt(
     &    'init: usage mmacos(''init'', model_size)')
        pr = mxGetPr(prhs(2))
        CALL mxCopyPtrToReal8(pr, buf, int8(1))
        model_size = int(buf(1))

        CALL init(ok, model_size)
        IF (.not. ok) CALL mexErrMsgTxt('mmacos: init failed')
        end subroutine do_init

C=======================================================================
C     load_rx                                                           =
C=======================================================================
        subroutine do_load_rx(nlhs, plhs, nrhs, prhs)
        use macos_api_mod, only: load_rx
        implicit none
        mwPointer plhs(*), prhs(*)
        integer nlhs, nrhs

        mwPointer mxGetM, mxGetN, mxCreateDoubleScalar
        integer*4 mxIsChar, mxGetString
        character(len=250) :: filename
        mwSize             :: slen
        integer*4          :: stat
        integer            :: n_elt_out
        logical            :: ok

        IF (nrhs .ne. 2) CALL mexErrMsgTxt(
     &    'load_rx: usage mmacos(''load_rx'', filename)')
        IF (mxIsChar(prhs(2)) .ne. 1) CALL mexErrMsgTxt(
     &    'load_rx: filename must be a string')

        slen = mxGetM(prhs(2)) * mxGetN(prhs(2))
        IF (slen .gt. len(filename)) CALL mexErrMsgTxt(
     &    'load_rx: filename too long')
        filename = ' '
        stat = mxGetString(prhs(2), filename, int(len(filename), 4))
        IF (stat .ne. 0) CALL mexErrMsgTxt(
     &    'load_rx: failed to read filename')

        CALL load_rx(ok, n_elt_out, filename)
        IF (.not. ok) CALL mexErrMsgTxt('mmacos: load_rx failed')

        IF (nlhs .ge. 1) THEN
          plhs(1) = mxCreateDoubleScalar(dble(n_elt_out))
        END IF
        end subroutine do_load_rx

C=======================================================================
C     save_rx                                                           =
C=======================================================================
        subroutine do_save_rx(nlhs, plhs, nrhs, prhs)
        use macos_api_mod, only: save_rx
        implicit none
        mwPointer plhs(*), prhs(*)
        integer nlhs, nrhs

        mwPointer mxGetM, mxGetN
        integer*4 mxIsChar, mxGetString
        character(len=250) :: filename
        integer*4          :: stat
        logical            :: ok

        IF (nrhs .ne. 2) CALL mexErrMsgTxt(
     &    'save_rx: usage mmacos(''save_rx'', filename)')
        IF (mxIsChar(prhs(2)) .ne. 1) CALL mexErrMsgTxt(
     &    'save_rx: filename must be a string')
        filename = ' '
        stat = mxGetString(prhs(2), filename, int(len(filename), 4))
        IF (stat .ne. 0) CALL mexErrMsgTxt(
     &    'save_rx: failed to read filename')
        CALL save_rx(ok, filename)
        IF (.not. ok) CALL mexErrMsgTxt('mmacos: save_rx failed')
        end subroutine do_save_rx

C=======================================================================
C     modified_rx                                                       =
C=======================================================================
        subroutine do_modified_rx(nlhs, plhs, nrhs, prhs)
        use macos_api_mod, only: modified_rx
        implicit none
        mwPointer plhs(*), prhs(*)
        integer nlhs, nrhs
        logical :: ok
        CALL modified_rx(ok)
        IF (.not. ok) CALL mexErrMsgTxt('mmacos: modified_rx failed')
        end subroutine do_modified_rx

C=======================================================================
C     n_elt                                                             =
C=======================================================================
        subroutine do_n_elt(nlhs, plhs, nrhs, prhs)
        use macos_api_mod, only: n_elt
        implicit none
        mwPointer plhs(*), prhs(*)
        integer nlhs, nrhs
        mwPointer mxCreateDoubleScalar
        integer :: nElt_out
        CALL n_elt(nElt_out)
        IF (nlhs .ge. 1) THEN
          plhs(1) = mxCreateDoubleScalar(dble(nElt_out))
        END IF
        end subroutine do_n_elt

C=======================================================================
C     opd  -- writes the current OPDMat (N x N where N=nGridPts) to    =
C             a MATLAB array.  N is queried via get_src_sampling -- the=
C             OPDMat dimension matches the live source-grid sampling,  =
C             not the static mpts compile-time limit.                   =
C=======================================================================
        subroutine do_opd(nlhs, plhs, nrhs, prhs)
        use macos_api_mod, only: opd_val, get_src_sampling
        implicit none
        mwPointer plhs(*), prhs(*)
        integer nlhs, nrhs

        mwPointer mxGetPr, mxCreateDoubleMatrix
        mwPointer pr_out
        integer   N
        real(8), allocatable :: opd_local(:,:)
        logical   ok
        integer*8 nbytes

        IF (nrhs .ne. 1) CALL mexErrMsgTxt(
     &    'opd: usage mmacos(''opd'')  (OPD comes from the last trace)')
        CALL get_src_sampling(ok, N)
        IF (.not. ok .or. N .le. 0) CALL mexErrMsgTxt(
     &    'mmacos: opd: Rx not loaded')
        allocate(opd_local(N, N))
        CALL opd_val(ok, opd_local, N)
        IF (.not. ok) THEN
          deallocate(opd_local)
          CALL mexErrMsgTxt('mmacos: opd_val failed (no prior trace?)')
        END IF
        IF (nlhs .ge. 1) THEN
          plhs(1) = mxCreateDoubleMatrix(
     &      int(N, kind=8), int(N, kind=8), 0)
          pr_out = mxGetPr(plhs(1))
          nbytes = int(N, kind=8) * int(N, kind=8)
          CALL mxCopyReal8ToPtr(opd_local, pr_out, nbytes)
        END IF
        deallocate(opd_local)
        end subroutine do_opd

C=======================================================================
C     intensity -- writes (mdttl x mdttl) intensity to MATLAB array     =
C=======================================================================
        subroutine do_intensity(nlhs, plhs, nrhs, prhs)
        use macos_api_mod, only: int_cmd, int_get
        implicit none
        mwPointer plhs(*), prhs(*)
        integer nlhs, nrhs

        mwPointer mxGetPr, mxCreateDoubleMatrix
        mwPointer pr_in, pr_out
        real(8)   buf(1)
        integer   iElt, N
        real(8), allocatable :: int_local(:,:)
        logical   ok, res_trace
        integer*8 nbytes

        IF (nrhs .lt. 2 .or. nrhs .gt. 3) CALL mexErrMsgTxt(
     &    'intensity: usage mmacos(''intensity'', iElt[, reset_trace])')
        pr_in = mxGetPr(prhs(2))
        CALL mxCopyPtrToReal8(pr_in, buf, int8(1))
        iElt = int(buf(1))
        res_trace = .true.
        IF (nrhs .ge. 3) THEN
          pr_in = mxGetPr(prhs(3))
          CALL mxCopyPtrToReal8(pr_in, buf, int8(1))
          res_trace = (buf(1) .ne. 0d0)
        END IF

        CALL int_cmd(ok, N, iElt, res_trace)
        IF (.not. ok .or. N .le. 0) CALL mexErrMsgTxt(
     &    'mmacos: intensity int_cmd failed')

        allocate(int_local(N, N))
        CALL int_get(ok, int_local, N)
        IF (.not. ok) THEN
          deallocate(int_local)
          CALL mexErrMsgTxt('mmacos: intensity int_get failed')
        END IF

        IF (nlhs .ge. 1) THEN
          plhs(1) = mxCreateDoubleMatrix(
     &      int(N, kind=8), int(N, kind=8), 0)
          pr_out = mxGetPr(plhs(1))
          nbytes = int(N, kind=8) * int(N, kind=8)
          CALL mxCopyReal8ToPtr(int_local, pr_out, nbytes)
        END IF
        deallocate(int_local)
        end subroutine do_intensity

C=======================================================================
C     complex_field -- writes (N x N) complex WFElt to MATLAB array     =
C=======================================================================
        subroutine do_complex_field(nlhs, plhs, nrhs, prhs)
        use macos_api_mod, only: cfield_cmd, cfield_get
        implicit none
        mwPointer plhs(*), prhs(*)
        integer nlhs, nrhs

        mwPointer mxGetPr
        mwPointer mxCreateDoubleMatrix
        mwPointer pr_in, pr_re, pr_im
        real(8)   buf(1)
        integer   iElt, N
        real(8), allocatable :: re_local(:,:), im_local(:,:)
        logical   ok, res_trace
        integer*8 nbytes

        IF (nrhs .lt. 2 .or. nrhs .gt. 3) CALL mexErrMsgTxt(
     &    'complex_field: usage mmacos(''complex_field'', iElt[, reset_trace])')
        pr_in = mxGetPr(prhs(2))
        CALL mxCopyPtrToReal8(pr_in, buf, int8(1))
        iElt = int(buf(1))
        res_trace = .true.
        IF (nrhs .ge. 3) THEN
          pr_in = mxGetPr(prhs(3))
          CALL mxCopyPtrToReal8(pr_in, buf, int8(1))
          res_trace = (buf(1) .ne. 0d0)
        END IF

        CALL cfield_cmd(ok, N, iElt, res_trace)
        IF (.not. ok .or. N .le. 0) CALL mexErrMsgTxt(
     &    'mmacos: complex_field cfield_cmd failed')

        allocate(re_local(N, N), im_local(N, N))
        CALL cfield_get(ok, re_local, im_local, N, iElt)
        IF (.not. ok) THEN
          deallocate(re_local, im_local)
          CALL mexErrMsgTxt('mmacos: complex_field cfield_get failed')
        END IF

        ! Return as a single complex matrix (MATLAB-side complex double).
        IF (nlhs .ge. 1) THEN
          plhs(1) = mxCreateDoubleMatrix(
     &      int(N, kind=8), int(N, kind=8), 1)
          pr_re = mxGetPr(plhs(1))
          nbytes = int(N, kind=8) * int(N, kind=8)
          ! With interleaved-complex (R2018a+), real/imag must be
          ! merged via mxComplexity flag.  Use mxCopyComplex16ToPtr
          ! when available; otherwise pack via mxGetPi + mxGetPr.
          ! For broad compatibility, use mxCopyPtrToComplex16-style:
          CALL mxCopyReal8ToPtr(re_local, pr_re, nbytes)
          ! Imag part goes via mxGetPi in legacy split layout, or
          ! interleaved via complex copy.  Use the universally-
          ! supported split layout: declare with mxComplexity=1
          ! (mxCOMPLEX) and write Pi.
          CALL write_imag(plhs(1), im_local, nbytes)
        END IF
        deallocate(re_local, im_local)
        end subroutine do_complex_field

C=======================================================================
C     dx_at                                                             =
C=======================================================================
        subroutine do_dx_at(nlhs, plhs, nrhs, prhs)
        use macos_api_mod, only: elt_dx_get
        implicit none
        mwPointer plhs(*), prhs(*)
        integer nlhs, nrhs

        mwPointer mxGetPr, mxCreateDoubleScalar
        mwPointer pr_in
        real(8)   buf(1), dx_out
        integer   iElt
        logical   ok

        IF (nrhs .ne. 2) CALL mexErrMsgTxt(
     &    'dx_at: usage mmacos(''dx_at'', iElt)')
        pr_in = mxGetPr(prhs(2))
        CALL mxCopyPtrToReal8(pr_in, buf, int8(1))
        iElt = int(buf(1))
        CALL elt_dx_get(ok, dx_out, iElt)
        IF (.not. ok) CALL mexErrMsgTxt('mmacos: dx_at failed')
        IF (nlhs .ge. 1) plhs(1) = mxCreateDoubleScalar(dx_out)
        end subroutine do_dx_at

C=======================================================================
C     base_unit_to_metres                                               =
C=======================================================================
        subroutine do_cbm(nlhs, plhs, nrhs, prhs)
        use macos_api_mod, only: base_unit_to_metres
        implicit none
        mwPointer plhs(*), prhs(*)
        integer nlhs, nrhs
        mwPointer mxCreateDoubleScalar
        real(8) :: cbm_out
        logical :: ok
        CALL base_unit_to_metres(ok, cbm_out)
        IF (.not. ok) CALL mexErrMsgTxt(
     &    'mmacos: base_unit_to_metres failed (Rx not loaded?)')
        IF (nlhs .ge. 1) plhs(1) = mxCreateDoubleScalar(cbm_out)
        end subroutine do_cbm

C=======================================================================
C     apodize -- multiply WFElt(:,:,iEltToiWF(iElt)) by real mask       =
C=======================================================================
        subroutine do_apodize(nlhs, plhs, nrhs, prhs)
        use macos_api_mod, only: cfield_apodize
        implicit none
        mwPointer plhs(*), prhs(*)
        integer nlhs, nrhs

        mwPointer mxGetPr, mxGetM, mxGetN
        mwPointer pr_in
        real(8)   buf(1)
        integer   iElt, N
        real(8), allocatable :: mask(:,:)
        logical   ok
        integer*8 nbytes

        IF (nrhs .ne. 3) CALL mexErrMsgTxt(
     &    'apodize: usage mmacos(''apodize'', iElt, mask)')
        pr_in = mxGetPr(prhs(2))
        CALL mxCopyPtrToReal8(pr_in, buf, int8(1))
        iElt = int(buf(1))
        N = int(mxGetM(prhs(3)))
        IF (mxGetN(prhs(3)) .ne. mxGetM(prhs(3))) CALL mexErrMsgTxt(
     &    'apodize: mask must be square')
        allocate(mask(N, N))
        pr_in = mxGetPr(prhs(3))
        nbytes = int(N, kind=8) * int(N, kind=8)
        CALL mxCopyPtrToReal8(pr_in, mask, nbytes)
        CALL cfield_apodize(ok, mask, N, iElt)
        deallocate(mask)
        IF (.not. ok) CALL mexErrMsgTxt('mmacos: apodize failed')
        end subroutine do_apodize

C=======================================================================
C     trace_rays -- run a full ray trace through the system; returns   =
C     [nRays, rms_WFE] (nRays = # successful rays, rms_WFE in WaveUnits)=
C=======================================================================
        subroutine do_trace_rays(nlhs, plhs, nrhs, prhs)
        use macos_api_mod, only: trace_rays
        implicit none
        mwPointer plhs(*), prhs(*)
        integer nlhs, nrhs

        mwPointer mxGetPr, mxCreateDoubleScalar
        mwPointer pr_in
        real(8)   buf(1), rms_WFE
        integer   iElt, nRays_out, N_out
        logical   ok

        IF (nrhs .ne. 2) CALL mexErrMsgTxt(
     &    'trace_rays: usage mmacos(''trace_rays'', iElt)')
        pr_in = mxGetPr(prhs(2))
        CALL mxCopyPtrToReal8(pr_in, buf, int8(1))
        iElt = int(buf(1))

        CALL trace_rays(ok, rms_WFE, nRays_out, N_out, iElt)
        IF (.not. ok) CALL mexErrMsgTxt('mmacos: trace_rays failed')

        IF (nlhs .ge. 1) plhs(1) = mxCreateDoubleScalar(dble(nRays_out))
        IF (nlhs .ge. 2) plhs(2) = mxCreateDoubleScalar(rms_WFE)
        end subroutine do_trace_rays

C=======================================================================
C     perturb_elt -- rigid-body PERTURB of one or more elements         =
C=======================================================================
        subroutine do_perturb_elt(nlhs, plhs, nrhs, prhs)
        use macos_api_mod, only: prb_elt
        implicit none
        mwPointer plhs(*), prhs(*)
        integer nlhs, nrhs

        mwPointer mxGetPr, mxGetM, mxGetN
        mwPointer pr_iElt, pr_prb, pr_glb
        integer   n_elt_arg, prb_rows, prb_cols, glb_n
        integer*8 nbytes
        real(8), allocatable :: iElt_d(:), prb_in(:,:), glb_d(:)
        integer, allocatable :: iElt_i(:), ifGlobal(:)
        real(8), allocatable :: prb(:,:)
        logical   ok
        integer   i

        IF (nrhs .ne. 4) CALL mexErrMsgTxt(
     &    'perturb_elt: usage mmacos(''perturb_elt'', iElt, prb_6xN, ifGlobal)')

        ! iElt: 1xN or Nx1 vector of element ids
        n_elt_arg = int(mxGetM(prhs(2)) * mxGetN(prhs(2)))
        IF (n_elt_arg .lt. 1) CALL mexErrMsgTxt(
     &    'perturb_elt: iElt must be non-empty')

        ! prb: 6xN matrix
        prb_rows = int(mxGetM(prhs(3)))
        prb_cols = int(mxGetN(prhs(3)))
        IF (prb_rows .ne. 6 .or. prb_cols .ne. n_elt_arg)
     &    CALL mexErrMsgTxt(
     &      'perturb_elt: prb must be 6 x len(iElt)')

        ! ifGlobal: Nx1 logical or numeric vector
        glb_n = int(mxGetM(prhs(4)) * mxGetN(prhs(4)))
        IF (glb_n .ne. n_elt_arg) CALL mexErrMsgTxt(
     &    'perturb_elt: ifGlobal must match length of iElt')

        allocate(iElt_d(n_elt_arg), iElt_i(n_elt_arg))
        pr_iElt = mxGetPr(prhs(2))
        CALL mxCopyPtrToReal8(pr_iElt, iElt_d, int(n_elt_arg, kind=8))
        iElt_i = int(iElt_d)

        allocate(prb_in(6, n_elt_arg), prb(6, n_elt_arg))
        pr_prb = mxGetPr(prhs(3))
        nbytes = int(6, kind=8) * int(n_elt_arg, kind=8)
        CALL mxCopyPtrToReal8(pr_prb, prb_in, nbytes)
        prb = prb_in

        allocate(glb_d(n_elt_arg), ifGlobal(n_elt_arg))
        pr_glb = mxGetPr(prhs(4))
        CALL mxCopyPtrToReal8(pr_glb, glb_d, int(n_elt_arg, kind=8))
        DO i = 1, n_elt_arg
          IF (glb_d(i) .ne. 0d0) THEN
            ifGlobal(i) = 1
          ELSE
            ifGlobal(i) = 0
          END IF
        END DO

        CALL prb_elt(ok, iElt_i, prb, ifGlobal, n_elt_arg)
        deallocate(iElt_d, iElt_i, prb_in, prb, glb_d, ifGlobal)
        IF (.not. ok) CALL mexErrMsgTxt('mmacos: perturb_elt failed')
        end subroutine do_perturb_elt

C=======================================================================
C     write_imag -- helper: write imag part to mxArray's pi pointer
C     This wraps the split-complex layout used pre-R2018a.  Newer
C     MATLABs prefer interleaved-complex via mxCreateDoubleMatrix
C     with mxComplexity=1 and direct interleaved write.  We use the
C     mxGetPi path which MATLAB keeps available for back-compat.
C=======================================================================
        subroutine write_imag(mxptr, im, n)
        implicit none
        mwPointer mxptr
        real(8)   im(*)
        integer*8 n
#if MX_HAS_INTERLEAVED_COMPLEX
        ! Interleaved layout (R2018a+): mxGetComplexDoubles then write.
        ! Build a complex array on the fly and use mxCopyComplex16ToPtr.
        mwPointer mxGetComplexDoubles
        complex(8), allocatable :: tmp(:)
        mwPointer pr_re
        mwPointer mxGetPr
        real(8), allocatable    :: re_buf(:)
        integer*8 :: i
        ! Read the real part we just wrote, repack into complex,
        ! call mxCopyComplex16ToPtr.
        allocate(re_buf(n), tmp(n))
        pr_re = mxGetPr(mxptr)
        CALL mxCopyPtrToReal8(pr_re, re_buf, n)
        DO i = 1, n
          tmp(i) = cmplx(re_buf(i), im(i), kind=8)
        END DO
        CALL mxCopyComplex16ToPtr(tmp, mxGetComplexDoubles(mxptr), n)
        deallocate(re_buf, tmp)
#else
        mwPointer mxGetPi
        mwPointer pi
        pi = mxGetPi(mxptr)
        IF (pi .ne. 0) THEN
          CALL mxCopyReal8ToPtr(im, pi, n)
        END IF
#endif
        end subroutine write_imag
