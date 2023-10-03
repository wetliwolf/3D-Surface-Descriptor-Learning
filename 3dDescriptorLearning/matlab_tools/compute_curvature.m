% function [Umin,Umax,Cmin,Cmax,Cmean,Cgauss,Normal] = compute_curvature(V,F)
function [Umax, Umin, Cmax, Cmin, Normal, hks, diameter] = compute_curvature(V, F, off_filename, hks_len)
% compute_curvature - compute principal curvature directions and values
%
%   [Umin,Umax,Cmin,Cmax,Cmean,Cgauss,Normal] = compute_curvature(V,F,options);
%
%   Umin is the direction of minimum curvature
%   Umax is the direction of maximum curvature
%   Cmin is the minimum curvature
%   Cmax is the maximum curvature
%   Cmean=(Cmin+Cmax)/2
%   Cgauss=Cmin*Cmax
%   Normal is the normal to the surface
%
%   options.curvature_smoothing controls the size of the ring used for
%       averaging the curvature tensor.
%
%   The algorithm is detailed in 
%       David Cohen-Steiner and Jean-Marie Morvan. 
%       Restricted Delaunay triangulations and normal cycle. 
%       In Proc. 19th Annual ACM Symposium on Computational Geometry, 
%       pages 237-246, 2003. 
%   and also in
%       Pierre Alliez, David Cohen-Steiner, Olivier Devillers, Bruno Le�vy, and Mathieu Desbrun. 
%       Anisotropic Polygonal Remeshing. 
%       ACM Transactions on Graphics, 2003. 
%       Note: SIGGRAPH '2003 Conference Proceedings
%
%   Copyright (c) 2007 Gabriel Peyre

orient = 1;

options.curvature_smoothing = 3;
options.null = 0;
naver = getoptions(options, 'curvature_smoothing', 3);


[V,F] = check_face_vertex(V,F);

n = size(V,2);
m = size(F,2);

diameter=shape_diameter(V',F');

% associate each edge to a pair of faces
i = [F(1,:) F(2,:) F(