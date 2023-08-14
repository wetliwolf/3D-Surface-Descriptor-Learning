
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
    void setSource(const Point& source, int face_idx);

    /**
     * Set source point on node node_idx.
     */
    void setNodeSource(int node_idx);

    /**
     * Set source on point, which lies on the face face_idx.
     */
    void setFaceSource(const Point& point, int face_idx);

    /**
     * Start generation of DGPC. When complete, distances and angles
     * for the nodes in a geodesic disk with radius stopdist_ will be
     * available with getDistance(ni) and getAngle(ni).
     */
    int run();

    /**
     * Get Gamma, the smallest ring of nodes connected by edges
     * surrounding the source point.  These nodes are initialized with
     * angles and distance after setSource is called.
     */
    const std::vector<int>& getGamma() { return gamma_; };

    /**
     * Get DGPC polar distance for node ni
     */
    real getDistance(int ni) { return distances_[ni]; };

    /**
     * Get DGPC polar angle for node ni
     */
    real getAngle(int ni) { return angles_[ni]; };

    /**
     * Get DGPC polar distances
     */
    const std::vector<real>& getDistances() { return distances_; };

    /**
     * Get DGPC polar angles
     */
    const std::vector<real>& getAngles() { return angles_; };


  protected:
    const Mesh& mesh_;
    real eps_;
    real stopdist_;

    GIGen::Heap<real> heap_;

    std::vector<real> distances_;
    std::vector<real> angles_;

    std::vector<int> gamma_;

    void initialize();

    real initializeGamma(const Point& point);

    bool tryComputeNodeFromEdge(int node, int edge[2]);
    real computeDistance(const Point& pt, int edge[2], real& alpha);
    real computeAngle(int node, int edge[2], real alpha);

  };

  //Implementation of setSource
  template<class Mesh>
    void
    Generator<Mesh>::setSource(const Point& point, int face_idx)
    {

      const real proximity_threshold = 10e-5;

      //Fetch nodes of the face
      std::vector<int> nodes;
      FaceHandle face = mesh_.face_handle(face_idx);
      HalfedgeHandle heh = mesh_.halfedge_handle(face);
      HalfedgeHandle start = heh;
      do {
        VertexHandle vh = mesh_.to_vertex_handle(heh);
        nodes.push_back(vh.idx());
        heh = mesh_.next_halfedge_handle(heh);
      } while(heh != start);

      //Is the source on a node?
      for(int i = 0; i < nodes.size(); i++) {
        VertexHandle vh = mesh_.vertex_handle(nodes[i]);
        const Point& np = mesh_.point(vh);
        
        if(np.dist(point) < proximity_threshold) {
          setNodeSource(nodes[i]);
          return;
        }
      }

      //Assume the source is on the face
      setFaceSource(point, face_idx);
      return;

    }

  //Implementation of setNodeSource
  template<class Mesh>
    void
    Generator<Mesh>::setNodeSource(int node_idx)
    {
      //Clear distances, angles and gamma
      initialize();

      //Initialize source node
      distances_[node_idx] = 0;
      angles_[node_idx] = 0;

      //Find gamma, walk along the 1-ring around source
      VertexHandle source = mesh_.vertex_handle(node_idx);
      HalfedgeHandle heh = mesh_.halfedge_handle(source);
      
      if(mesh_.is_boundary(source)) {
        //Skip anticlockwise around source until heh is the last non-boundary halfedge
        HalfedgeHandle b = mesh_.opposite_halfedge_handle(heh);
        while(!mesh_.is_boundary(b)) {
          heh = mesh_.next_halfedge_handle(b);
          b = mesh_.opposite_halfedge_handle(heh);
        }
      }

      HalfedgeHandle start = heh;
      VertexHandle to;

      //Traverse all halfedges pointing into source
      do {
        heh = mesh_.next_halfedge_handle(heh);
        to = mesh_.to_vertex_handle(heh);

        //Traverse all nodes on the edge of this face, except source and first anticlockwise neighhbour
        while (to != source) {
          gamma_.push_back(to.idx());
          heh = mesh_.next_halfedge_handle(heh);
          to = mesh_.to_vertex_handle(heh);
        } 
        //heh is now pointing to source
        heh = mesh_.opposite_halfedge_handle(heh);

      } while(heh != start);

      Point source_pt = mesh_.point(source);

      //Initialize gamma with distances and angles
      real phitot = initializeGamma(source_pt);

      if(!mesh_.is_boundary(source)) {
        //Scale angles to sum to 2pi
        const real alpha = (2*M_PI)/phitot;
        const int num = gamma_.size();
        for(unsigned int i = 0; i < num; i++) {
          //Store the angle for this node
          angles_[gamma_[i]] *= alpha;
        }
      }

    }

  //Implementation of setFaceSource
  template<class Mesh>
    void
    Generator<Mesh>::setFaceSource(const Point& point, int face_idx)
    {

      //Clear distances, angles and gamma
      initialize();

      //Find gamma, the nodes of this face.
      FaceHandle face = mesh_.face_handle(face_idx);
      HalfedgeHandle heh = mesh_.halfedge_handle(face);
      HalfedgeHandle start = heh;
      do {
        VertexHandle vh = mesh_.to_vertex_handle(heh);
        int ni = vh.idx();
        gamma_.push_back(ni);
        heh = mesh_.next_halfedge_handle(heh);
      } while(heh != start);

      //Initialize gamma with distances and angles
      initializeGamma(point);
    }


  //Implementation of run
  template<class Mesh>
    int
    Generator<Mesh>::run()
    {

      int last_finished = -1;

      int edges[3];
      std::vector<int> next;

      HalfedgeHandle heh, end;

      while (!heap_.empty()) {
    
        int curr = heap_.getCandidate();
        if (curr == -1) break;
    
        last_finished = curr;

        VertexHandle curr_vertex = mesh_.vertex_handle(curr);

        //Iterate halfedges pointing into current vertex (one for each
        //face adjacent to curr)
        HalfedgeHandle face_start = mesh_.halfedge_handle(curr_vertex);
        HalfedgeHandle face = face_start;

        do {
          face = mesh_.opposite_halfedge_handle(face);
          if(!mesh_.is_boundary(face)) {
            heh = mesh_.prev_halfedge_handle(face);
            end = heh;

            //For this face, we will attempt to compute DGPC from each
            //of the two edges connected to source
            edges[0] = mesh_.to_vertex_handle(heh).idx();
            heh = mesh_.next_halfedge_handle(heh);