function D = afocal4_seed(P, opts)
%AFOCAL4_SEED  The design struct the S4 ladder starts from.
%
%   D = AFOCAL4_SEED(P) is the S3-recommended layout as an AFOCAL4_BUILD
%   design: the parent's conics carried, the new field mirror at K = 0, the
%   field-mirror standoff at P.fm_standoff, the interface at P.iface, no
%   rigid body, at the design bias.
%
%   THE CONICS ARE THE PARENT'S AND THAT IS A SEED, NOT A CLAIM.  The
%   closure moves the collimator's radius by ~29% at the flagged operating
%   point (phi4 = +2 /m), so the carried conics are wrong for the layout they
%   are carried into -- which is exactly why the unoptimised S3 layouts read
%   14 um of wavefront error.  The seed exists to be left behind.
%
%   D = AFOCAL4_SEED(P, 'form','mersenne') seeds the hedge instead: the
%   double Mersenne's four parabolas, confocal spacings held.
%
%   Name-value:  'form' ('field'|'mersenne'), 'bias_deg', 'iface',
%                'fm_standoff', 'ngrid'
%
%   See also AFOCAL4_BUILD, AFOCAL4_SOLVE.

    arguments
        P (1,1) struct
        opts.form        (1,:) char = P.form
        opts.bias_deg    (1,1) double = P.bias_deg
        opts.iface       (1,1) double = P.iface
        opts.fm_standoff (1,1) double = P.fm_standoff
        opts.ngrid       (1,1) double = P.ngrid
    end

    D = struct('form',opts.form, 'iface',opts.iface, ...
               'fm_standoff',opts.fm_standoff, 'gap',P.mersenne.gap, ...
               'bias_deg',opts.bias_deg, 'ngrid',opts.ngrid, ...
               'R2',P.parent.R(2), 't1',P.parent.t(1), ...
               'rb', zeros(numel(P.rb_elts),2));

    switch opts.form
    case 'field'
        % build order M1, M2, FM, M3 -- his three carried, the new one flat
        D.K = [P.parent.K(1), P.parent.K(2), 0, P.parent.K(3)];
    case 'mersenne'
        D.K = [-1 -1 -1 -1];
    otherwise
        error('macos:design:afocal4_seed:form', 'unknown form "%s".', opts.form);
    end
end
