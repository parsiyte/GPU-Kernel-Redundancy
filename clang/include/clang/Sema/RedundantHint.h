#ifndef LLVM_CLANG_SEMA_RedundantHint_H
#define LLVM_CLANG_SEMA_RedundantHint_H

#include "clang/Basic/IdentifierTable.h"
#include "clang/Basic/LLVM.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Sema/Ownership.h"
#include "clang/Sema/ParsedAttr.h"
#include "clang/Tooling/Inclusions/IncludeStyle.h"
#include "llvm/ADT/StringRef.h"
#include <string>


namespace clang {
struct RedundantHint {
  SourceRange Range;
  IdentifierLoc *PragmaNameLoc;
  IdentifierLoc *OptionLoc;
  Expr *Inputs;
  Expr *Outputs;
  IdentifierLoc *SchemeType;

  RedundantHint()
      : PragmaNameLoc(nullptr), OptionLoc(nullptr), 
      Inputs(nullptr),
      Outputs(nullptr) {}
};

}
#endif // LLVM_CLANG_SEMA_RedundantHint_H
