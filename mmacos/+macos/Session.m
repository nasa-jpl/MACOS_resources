classdef Session < handle
%MACOS.SESSION  OO veneer over the +macos function package.
%
%   m = macos.Session(model_size) initializes the engine and returns a
%   handle.  Every method delegates to the same-named +macos package
%   function -- the only state in the class is the loaded Rx path
%   (cached so it's discoverable via m.rx_path).
%
%   This is purely a notational convenience.  Both styles work and
%   share the same underlying libsmacos.a state:
%
%     m = macos.Session(); m.load_rx('foo'); W = m.opd();
%     macos.init(256);     macos.load_rx('foo'); W = macos.opd();
%
%   Use the class when MATLAB code reads more naturally with dot
%   notation (e.g. inside a method or function that "owns" a macos
%   session); use the package functions when scripting or when one
%   call site doesn't need a handle.

    properties (SetAccess = private)
        model_size
        rx_path = ''
    end

    methods
        function obj = Session(model_size)
            arguments
                model_size (1,1) double {mustBeInteger, mustBePositive} = 256
            end
            macos.init(model_size);
            obj.model_size = model_size;
        end

        % --- Rx lifecycle ---------------------------------------------
        function n = load_rx(obj, rx_path)
            n = macos.load_rx(rx_path);
            obj.rx_path = rx_path;
        end
        function save_rx(obj, rx_path), macos.save_rx(rx_path); end
        function modify(obj),           macos.modify();          end
        function n = num_elt(obj),      n = macos.num_elt();     end
        function tf = has_rx(obj),      tf = macos.has_rx();     end

        % --- System / units -------------------------------------------
        function c = cbm(obj),          c = macos.cbm();         end
        function s = sys_units(obj),    s = macos.sys_units();   end

        % --- Source ---------------------------------------------------
        function n = get_src_sampling(obj), n = macos.get_src_sampling(); end
        function set_src_sampling(obj, n),  macos.set_src_sampling(n);    end
        function w = get_src_wvl(obj),      w = macos.get_src_wvl();      end
        function set_src_wvl(obj, w),       macos.set_src_wvl(w);         end

        % --- Element geometry -----------------------------------------
        function v = get_elt_vpt(obj, srf), v = macos.get_elt_vpt(srf); end
        function set_elt_vpt(obj, srf, v),  macos.set_elt_vpt(srf, v);  end
        function p = get_elt_psi(obj, srf), p = macos.get_elt_psi(srf); end
        function set_elt_psi(obj, srf, p),  macos.set_elt_psi(srf, p);  end
        function r = get_elt_rpt(obj, srf), r = macos.get_elt_rpt(srf); end
        function set_elt_rpt(obj, srf, r),  macos.set_elt_rpt(srf, r);  end

        % --- Perturbations --------------------------------------------
        function perturb(obj, srf, varargin)
            macos.perturb(srf, varargin{:});
        end
        function perturb_many(obj, srf_vec, prb, is_global)
            macos.perturb_many(srf_vec, prb, is_global);
        end
        function perturb_src(obj, varargin)
            macos.perturb_src(varargin{:});
        end

        % --- Trace + diffraction buffers ------------------------------
        function s = trace(obj, srf)
            if nargin < 2
                s = macos.trace();
            else
                s = macos.trace(srf);
            end
        end
        function W = opd(obj),          W = macos.opd();             end
        function I = intensity(obj, srf, varargin)
            I = macos.intensity(srf, varargin{:});
        end
        function c = complex_field(obj, srf, varargin)
            c = macos.complex_field(srf, varargin{:});
        end
        function dx = dx_at(obj, srf, unit)
            if nargin < 3, unit = 'm'; end
            dx = macos.dx_at(srf, unit);
        end
        function apodize(obj, srf, mask)
            macos.apodize(srf, mask);
        end

        % --- Field-of-view / stop / exit pupil ------------------------
        function f = get_src_fov(obj),  f = macos.get_src_fov();    end
        function set_src_fov(obj, varargin)
            macos.set_src_fov(varargin{:});
        end
        function stop(obj, iElt, vpt_offset)
            if nargin < 3
                macos.stop(iElt);
            else
                macos.stop(iElt, vpt_offset);
            end
        end
        function stop_obj(obj, x, y, z),  macos.stop_obj(x, y, z);   end
        function xp = sxp(obj, mode)
            if nargin < 2
                xp = macos.sxp();
            else
                xp = macos.sxp(mode);
            end
        end
        function srs(obj, iSlv1, iSlv2, varargin)
            macos.srs(iSlv1, iSlv2, varargin{:});
        end
        function s = get_elt_csys(obj, srfs)
            s = macos.get_elt_csys(srfs);
        end

        % --- Sensitivity-channel eligibility --------------------------
        function ff = find_freeform_elts(obj)
            ff = macos.find_freeform_elts();
        end
        function ze = find_zern_elts(obj, rx_path)
            if nargin < 2, rx_path = obj.rx_path; end
            ze = macos.find_zern_elts(rx_path);
        end

        % --- Element Zernike coefficients -----------------------------
        function set_elt_zrn_coef(obj, iElt, modes, coefs, varargin)
            macos.set_elt_zrn_coef(iElt, modes, coefs, varargin{:});
        end
        function c = get_elt_zrn_coef(obj, iElt, modes)
            c = macos.get_elt_zrn_coef(iElt, modes);
        end
        function set_elt_mon_zrn_coef(obj, iElt, modes, coefs, varargin)
            macos.set_elt_mon_zrn_coef(iElt, modes, coefs, varargin{:});
        end
        function c = get_elt_mon_zrn_coef(obj, iElt, modes)
            c = macos.get_elt_mon_zrn_coef(iElt, modes);
        end
        function set_elt_ff_zrn_coef(obj, iElt, modes, coefs, varargin)
            macos.set_elt_ff_zrn_coef(iElt, modes, coefs, varargin{:});
        end
        function c = get_elt_ff_zrn_coef(obj, iElt, modes)
            c = macos.get_elt_ff_zrn_coef(iElt, modes);
        end

        % --- Element groups (EltGrp / GPERTURB) -----------------------
        function set_elt_grp(obj, iElt, members)
            macos.set_elt_grp(iElt, members);
        end
        function m = get_elt_grp(obj, iElt),  m = macos.get_elt_grp(iElt); end
        function del_elt_grp(obj, iElt),      macos.del_elt_grp(iElt);     end
        function prb_grp(obj, iElt, prb, ifGlobal)
            if nargin < 4
                macos.prb_grp(iElt, prb);
            else
                macos.prb_grp(iElt, prb, ifGlobal);
            end
        end
    end
end
