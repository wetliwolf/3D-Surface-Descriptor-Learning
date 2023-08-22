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
#ifndef DGPC_HEAP_H
#define DGPC_HEAP_H

#include <queue>
#include <functional>

namespace GIGen {

template<class real>
class HeapNode {

public:
  int   idx_;
  real  key_;

  HeapNode( int idx, real key) { idx_ = idx; key_ = key;}
  ~HeapNode(){}

  bool operator >  ( const HeapNode<real>& x) const { return (this->key_ >  x.key_);}
  bool operator >= ( const HeapNode<real>& x) const { return (this->key_ >= x.key_);}
  bool operator <  ( const HeapNode<real>& x) const { return (this->key_ <  x