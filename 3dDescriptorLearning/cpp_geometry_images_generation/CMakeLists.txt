cmake_minimum_required(VERSION 3.6)

project(GIGen)

# Note, replace these directories with the correct paths on your system
include_directories( ${CMAKE_CURRENT_LIST_DIR}\\include )
include_directories( C:\\dev\\OpenMesh-6.3\\include ) # OpenMesh head files directory. 
include_directories( C:\\Program\ Files\\MATLAB\\MATLAB\ Runtime\\v93\\extern\\include ) # MATLAB Runtime head files directory. 

link_directories( ${CMAKE_CURRENT_LIST_DIR}\\lib ) 
link_directories( C:\\dev\\Op