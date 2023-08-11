
/************************************************************
 * This file is part of the DGPC library. The library computes
 * Discrete Geodesic Polar Coordinates on a polygonal mesh.
 *
 * More info:
 *   http://folk.uio.no/eivindlm/dgpc/
 *
 * Authors: Eivind Lyche Melvær and Martin Reimers
 * Centre of Mathematics and Department of Informatics
 * University of Oslo, Norway, 2012
 ************************************************************/
#ifndef DGPC_GENERATOR_H
#define DGPC_GENERATOR_H

#include "Heap.h"
#include "Mesh_C.h"

#include <limits>
#include <algorithm>

namespace GIGen {

  /**
   * A class to generate DGPC: Discrete Geodesic Polar Coordinates on
   * polygonal meshes. The class is templetized with a halfedge mesh
   * datastructure, and has been tested with OpenMesh. It should
   * probably also work on a CGAL::Polyhedron_3 without too much
   * effort.
   */
  template<class Mesh>
    class Generator {

    typedef typename Mesh::point_type Point;
    typedef typename Point::value_type real;

    typedef typename Mesh::FaceHandle FaceHandle;
    typedef typename Mesh::HalfedgeHandle HalfedgeHandle;
    typedef typename Mesh::VertexHandle VertexHandle;

  public:
    /**
     * Construct a Generator for a Mesh.
     */
  Generator(const Mesh& mesh) :
    mesh_(mesh) {
      eps_ = 1e-12;
      stopdist_ = (std::numeric_limits<real>::max)();
      const int n = mesh_.n_vertices();
      distances_.resize(n);
      angles_.resize(n);
    };

    /**
     * Set epsilon. The algorithm will skip iterations which are not
     * significant for accuracy less than the given epsilon.
     */
    void setEps(real eps) { eps_ = eps; };
    /**
     * Set stop distance, geodesic radius of patch to compute.
     */
    void setStopDist(real d) { stopdist_ = d; };

    /**
     * Set source point. The point is assumed to be either one of the
     * nodes of face_idx, or on one of the faces.
     */