#ifndef LLVM_CLANG_SEMA_QUALITYHINT_H
#define LLVM_CLANG_SEMA_QUALITYHINT_H

#include "clang/Basic/IdentifierTable.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Sema/Ownership.h"
#include "clang/Sema/ParsedAttr.h"

namespace clang {
struct QualityHint {
  SourceRange Range;
  IdentifierLoc *PragmaNameLoc;
  IdentifierLoc *OptionLoc;
  Expr *ValueExprF;
  Expr *ValueExpr;

  QualityHint()
      : PragmaNameLoc(nullptr), OptionLoc(nullptr), ValueExprF(nullptr), ValueExpr(nullptr) {}
};

}
#endif // LLVM_CLANG_SEMA_QUALITYHINT_H
