#ifndef LLVM_CLANG_SEMA_RedundantHint_H
#define LLVM_CLANG_SEMA_RedundantHint_H

#include "clang/Basic/IdentifierTable.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Sema/Ownership.h"
#include "clang/Sema/ParsedAttr.h"

namespace clang {
struct RedundantHint {
  SourceRange Range;
  IdentifierLoc *PragmaNameLoc;
  IdentifierLoc *OptionLoc;
  Expr *ValueExprF;
  Expr *ValueExpr;

  RedundantHint()
      : PragmaNameLoc(nullptr), OptionLoc(nullptr), ValueExprF(nullptr), ValueExpr(nullptr) {}
};

}
#endif // LLVM_CLANG_SEMA_RedundantHint_H
