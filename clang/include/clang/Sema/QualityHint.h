#ifndef LLVM_CLANG_SEMA_QUALITYHINT_H
#define LLVM_CLANG_SEMA_QUALITYHINT_H

#include "clang/Basic/IdentifierTable.h"
#include "clang/Basic/LLVM.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Sema/Ownership.h"
#include "clang/Sema/ParsedAttr.h"
#include "llvm/ADT/StringRef.h"
#include <string>

namespace clang {
struct QualityHint {
  SourceRange Range;
  IdentifierLoc *PragmaNameLoc;
  IdentifierLoc *OptionLoc;
  Expr *Inputs;
  Expr *Inputs2;
  Expr *Inputs3;
  Expr *Inputs4;
  Expr *Inputs5;
  Expr *Outputs;
  Expr *Outputs2;
  Expr *Outputs3;
  Expr *Outputs4;
  Expr *Outputs5;

  QualityHint()
      : PragmaNameLoc(nullptr), OptionLoc(nullptr), 
      Inputs(nullptr), Inputs2(nullptr), Inputs3(nullptr), Inputs4(nullptr), Inputs5(nullptr), 
      Outputs(nullptr), Outputs2(nullptr), Outputs3(nullptr), Outputs4(nullptr), Outputs5(nullptr) {}
};

}
#endif // LLVM_CLANG_SEMA_QUALITYHINT_H
