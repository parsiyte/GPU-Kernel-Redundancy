# If we don't need RTTI or EH, there's no reason to export anything
# from the duplicate plugin.
if( NOT LLVM_REQUIRES_RTTI )
  if( NOT LLVM_REQUIRES_EH )
    set(LLVM_EXPORTED_SYMBOL_FILE ${CMAKE_CURRENT_SOURCE_DIR}/RMTv2.exports)
  endif()
endif()

if(WIN32 OR CYGWIN)
  set(LLVM_LINK_COMPONENTS Core Support)
endif()

add_llvm_library( LLVMRMTv2 MODULE BUILDTREE_ONLY
  RMTv2.cpp
  DEPENDS
  intrinsics_gen
  PLUGIN_TOOL
  opt
  )
