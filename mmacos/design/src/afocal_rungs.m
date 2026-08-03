function [v, W, info] = afocal_rungs(P, D, L, p1, d1, a)
%AFOCAL_RUNGS  The three reference-freedom rungs of the AFOCAL WFE metric.
%
%   v = AFOCAL_RUNGS(P, D, L, p1, d1, a) returns a 1x3 row of RMS wavefront
%   errors in METRES, one per rung, for ONE field.  Arguments are the ray
%   arrays AFOCAL_REFS takes: P (3,N) positions at the reference surface,
%   D (3,N) directions there, L (N) cumulative OPL, p1/d1 the exit chief,
%   a (3,1) the reference anchor (the coldstop vertex).
%
%   [v, W] = ... also returns the per-ray RESIDUAL WAVEFRONTS, W (N,3), one
%   column per rung, each already reduced by that rung's freedoms (piston
%   is left in; every rung's statistic is a std, so piston never enters).
%   Hand a column to a Strehl evaluation -- |mean(exp(i*2*pi*W/lambda))|^2
%   is the exact aperture form and needs the wavefront, not its RMS.
%
%   [v, W, info] = ... also returns the AFOCAL_REFS struct, i.e. the
%   physical quantities the freedoms correspond to.
%
%   THE RUNGS, in order of increasing reference freedom:
%     1  piston-only    flat reference normal to the exit CHIEF ray.  The
%                       honest number: everything the system did to the
%                       wavefront, including its pointing.
%     2  + tip/tilt     least-squares tip/tilt removed over the pupil.  The
%                       removed term is a BORESIGHT, not an error -- an
%                       afocal telescope's job is to hand its instrument a
%                       collimated beam, and which way that beam points is
%                       the pointing budget's problem, not the wavefront's.
%                       Reported as .tilt_urad.
%     3  + power        rung 2 with defocus removed as well.  The removed
%                       term is the RESIDUAL DIVERGENCE -- the output is not
%                       quite collimated.  Reported BOTH as a wavefront sag
%                       (.power_sag_nm) and as an angle (.divergence_urad),
%                       because a collimation error is naturally an angle
%                       and quoting it only in nm hides how big it is.
%
%   The rungs are ORDERED: each is the previous one with a further
%   least-squares term removed, so rung k+1 <= rung k EXACTLY, by
%   construction rather than by a solver behaving well.  (Contrast
%   STRICT_RUNGS, whose focus rung is a bounded search and needed an
%   explicit ff(0) guard.)  Always name the rung a quoted number came from.
%
%   THERE IS NO FOCUS RUNG.  A focal system's detector can slide along the
%   chief and buy back defocus; an afocal system has no detector, so the
%   equivalent freedom is the collimation of the delivered beam -- rung 3
%   -- and it is a DELIVERABLE, not a free choice.  Quote rung 1 or 2 as
%   the headline and rung 3 only alongside .divergence_urad.
%
%   See also AFOCAL_REFS, AFOCAL_PLANE_OPL, AFOCAL_LADDER_DECK,
%   STRICT_RUNGS.

    f = afocal_refs(P, D, L, p1, d1, a);
    A1 = [ones(numel(f.px),1), f.px, f.py];
    A2 = [A1, f.px.^2 + f.py.^2];

    W1 = f.W;
    W2 = W1 - A1*(A1\W1);
    W3 = W1 - A2*(A2\W1);

    v = [std(W1), std(W2), std(W3)];
    if nargout > 1, W = [W1, W2, W3]; end
    if nargout > 2
        info = f;
        info.power_sag_nm = f.power_sag_m*1e9;
    end
end
