% function [Umin,Umax,Cmin,Cmax,Cmean,Cgauss,Normal] = compute_curvature(V,F)
function [Umax, Umin, Cmax, Cmin, Normal, hks, diameter] = compute_curvature(V, F, off_filename, hks_len)
% compute_curvature - compute principal curvature directions and values
%
%   [Umin,Uma