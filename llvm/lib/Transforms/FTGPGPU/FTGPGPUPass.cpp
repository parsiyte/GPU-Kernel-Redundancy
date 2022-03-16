#ifndef MYHEADEFILE_H2
#define MYHEADEFILE_H2
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/raw_ostream.h"
#include <cstddef>
#include <string>
#include "common.h"

using namespace llvm;

  
class AbstractPass{
public:
    virtual ~AbstractPass() {}
    virtual bool executeThePass(CallInst* FunctionCall, Auxiliary* PassAuxiliary) const = 0;
};


class FTGPGPUPass{
  private:
  AbstractPass* Pass;

  public:
    FTGPGPUPass(CallInst* FunctionCall, Auxiliary* PassAuxiliary, AbstractPass * Pass = nullptr) : Pass(Pass){this->FunctionCall = FunctionCall; this->PassAuxiliary = PassAuxiliary;}
    ~FTGPGPUPass() { delete this->Pass;}
    CallInst* FunctionCall;
    Auxiliary* PassAuxiliary;
    void setPass (AbstractPass *Pass){
      delete  this->Pass;
      this->Pass = Pass;
    }
    void runThePass(){
      this->Pass->executeThePass(this->FunctionCall, this->PassAuxiliary);
    }

};

#endif


