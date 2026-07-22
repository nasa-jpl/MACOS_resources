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
        function s = spot(obj, srf, varargin)
            s = macos.spot(srf, varargin{:});
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

        % --- FreeForm composite Zernike + Grid -----------------------
        function s = zrn_freeform(obj, srf, varargin)
            if nargout > 0
                s = macos.zrn_freeform(srf, varargin{:});
            else
                macos.zrn_freeform(srf, varargin{:});
            end
        end
        function elt_grid_add(obj, srf, grid_dz)
            macos.elt_grid_add(srf, grid_dz);
        end

        % --- CALIB design optimizer -----------------------------------
        function r = calib(obj)
            r = macos.calib();
        end
        function calib_clear_var_elts(obj)
            macos.calib_clear_var_elts();
        end
        function calib_set_var_elt(obj, srf, varargin)
            macos.calib_set_var_elt(srf, varargin{:});
        end
        function calib_set_iter(obj, n_iter)
            macos.calib_set_iter(n_iter);
        end
        function calib_set_tol(obj, tol)
            macos.calib_set_tol(tol);
        end
        function calib_set_target(obj, target, varargin)
            macos.calib_set_target(target, varargin{:});
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

        % --- Element conic / radius / geometry queries ----------------
        function kc = get_elt_kc(obj, srf),      kc = macos.get_elt_kc(srf);   end
        function set_elt_kc(obj, srf, kc),       macos.set_elt_kc(srf, kc);     end
        function kr = get_elt_kr(obj, srf),      kr = macos.get_elt_kr(srf);   end
        function set_elt_kr(obj, srf, kr),       macos.set_elt_kr(srf, kr);     end
        function z  = get_elt_z(obj, srf),       z  = macos.get_elt_z(srf);    end
        function i  = get_elt_info(obj, srf),    i  = macos.get_elt_info(srf); end
        function o  = get_elt_obs(obj, srf),     o  = macos.get_elt_obs(srf);  end

        % --- Element surface inspection (grid / Zernike / csys) -------
        function tf = elt_grid_any(obj),         tf = macos.elt_grid_any();     end
        function tf = elt_zrn_any(obj),          tf = macos.elt_zrn_any();      end
        function tf = elt_ff_any(obj),           tf = macos.elt_ff_any();       end
        function n  = grid_size_max(obj),        n  = macos.grid_size_max();    end
        function n  = mon_zrn_max_modes(obj),    n  = macos.mon_zrn_max_modes();end
        function g  = get_elt_grid(obj, srf),    g  = macos.get_elt_grid(srf);  end
        function set_elt_grid(obj, srf, dx, mat),macos.set_elt_grid(srf, dx, mat); end
        function scale_elt_grid(obj, srf, f),    macos.scale_elt_grid(srf, f);  end
        function n  = get_elt_grid_size(obj, srf),    n  = macos.get_elt_grid_size(srf);    end
        function dx = get_elt_grid_spacing(obj, srf), dx = macos.get_elt_grid_spacing(srf); end
        function set_elt_grid_spacing(obj, srf, dx),  macos.set_elt_grid_spacing(srf, dx);  end
        function z  = get_elt_zrn(obj, srf),          z  = macos.get_elt_zrn(srf);          end
        function t  = get_elt_zrn_type(obj, srf),     t  = macos.get_elt_zrn_type(srf);     end
        function set_elt_zrn_type(obj, srf, varargin),macos.set_elt_zrn_type(srf, varargin{:}); end
        function r  = get_elt_zrn_norm_radius(obj, srf),   r = macos.get_elt_zrn_norm_radius(srf); end
        function set_elt_zrn_norm_radius(obj, srf, r),     macos.set_elt_zrn_norm_radius(srf, r);  end
        function s  = get_elt_srf_csys(obj, srfs),    s  = macos.get_elt_srf_csys(srfs);    end
        function set_elt_srf_csys(obj, srf, p, x, y, z),   macos.set_elt_srf_csys(srf, p, x, y, z); end
        function set_elt_csys(obj, srf, x, y, z, varargin),macos.set_elt_csys(srf, x, y, z, varargin{:}); end
        function rm_elt_csys(obj, srfs),              macos.rm_elt_csys(srfs);              end

        % --- Grating inspection ---------------------------------------
        function tf = elt_grating_any(obj),      tf = macos.elt_grating_any();  end
        function s  = elt_grating_fnd(obj, srfs),s  = macos.elt_grating_fnd(srfs); end
        function r  = get_elt_grating_type(obj, srf),  r = macos.get_elt_grating_type(srf);  end
        function o  = get_elt_grating_order(obj, srf), o = macos.get_elt_grating_order(srf); end
        function d  = get_elt_grating_dir(obj, srf),   d = macos.get_elt_grating_dir(srf);   end
        function w  = get_elt_grating_rulewidth(obj, srf), w = macos.get_elt_grating_rulewidth(srf); end
        function p  = get_elt_grating_params(obj, srf),    p = macos.get_elt_grating_params(srf);    end

        % --- Ray-trace status queries ---------------------------------
        function r = get_ray_info(obj, n),       r = macos.get_ray_info(n);     end
        function r = get_ray_status(obj, n),     r = macos.get_ray_status(n);   end

        % --- Stop / exit-pupil / first-order --------------------------
        function s = get_stop_info(obj),         s = macos.get_stop_info();     end
        function s = fex(obj, varargin),         s = macos.fex(varargin{:});    end
        function ffp(obj, varargin),             macos.ffp(varargin{:});         end
        function pfp(obj, varargin),             macos.pfp(varargin{:});         end
        function xps(obj, iElt),                 macos.xps(iElt);                end
        function s = get_xp(obj),                s = macos.get_xp();            end
        function set_xp(obj, vpt, psi, rad),     macos.set_xp(vpt, psi, rad);    end
        function p = first_order_properties(obj, varargin)
            p = macos.first_order_properties(varargin{:});
        end
        function pq = pupil_quality(obj, ep_elt, varargin)
            pq = macos.pupil_quality(ep_elt, varargin{:});
        end

        % --- Source queries -------------------------------------------
        function s = get_src_size(obj),          s = macos.get_src_size();      end
        function set_src_size(obj, varargin),    macos.set_src_size(varargin{:}); end
        function tf = is_point_source(obj),      tf = macos.is_point_source();  end
        function s = get_src_csys(obj),          s = macos.get_src_csys();      end
        function f = get_src_flux(obj),          f = macos.get_src_flux();      end
        function set_src_flux(obj, flux),        macos.set_src_flux(flux);       end

        % --- Diffraction buffers / window -----------------------------
        function I = compose(obj, srf, varargin), I = macos.compose(srf, varargin{:}); end
        function window(obj, varargin),          macos.window(varargin{:});      end
        function window_off(obj),                macos.window_off();             end

        % --- Element finders ------------------------------------------
        function g = find_grid_elts(obj),        g = macos.find_grid_elts();    end
    end
end
