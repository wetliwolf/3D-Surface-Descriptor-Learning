function f = perform_mesh_smoothing(face,vertex,f,options)

% perform_mesh_smoothing - smooth a function defined on a mesh by averaging
%
%   f = perform_mesh_smoothing(face,vertex,f,options);
%
%   Smooth a function f on a width of options.niter_averaging vertices.
%
%   Copyright (c) 2007 Gabriel Peyre

options.null = 0;
naver = getoptions(options, 'niter_averaging', 1);
type = getoptions(options, 'averaging_type', 'combinatorial');

if nargin<3
    f 