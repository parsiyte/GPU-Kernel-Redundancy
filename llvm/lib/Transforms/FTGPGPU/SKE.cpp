#include "llvm/ADT/StringRef.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/raw_ostream.h"
#include <cstddef>
#include <string>


using namespace llvm;


class SKE : public AbstractPass{
  
  public:
  
    SKE() {}
    
    bool executeThePass(CallInst* FunctionCall, Auxiliary* PassAuxiliary) const override{

      BasicBlock* CurrentBB = FunctionCall->getParent();
      BasicBlock* NextBB = CurrentBB->getNextNode();
      BasicBlock* PrevBB = CurrentBB->getPrevNode();
      Value* StreamArray;
      Function* FunctionToReplicate = FunctionCall->getCalledFunction();
      FunctionCallee StreamCreateFunction = PassAuxiliary->StreamCreateFunction;

      Output* OutputObject = PassAuxiliary->OutputObject;
      bool IsLoop = isForLoop(PrevBB);

      Instruction* FirstInstructionOfNextBB = NextBB->getFirstNonPHI();
      Instruction* FirstInstructionOfPrevBB = PrevBB->getFirstNonPHI();
      Instruction& LastInstructionOfPrevBB = PrevBB->back();
      IRBuilder<> Builder(FunctionCall); 
      Value* Zero32bit = PassAuxiliary->Zero32Bit;
      Value* One32bit = PassAuxiliary->One32Bit;
      Type* Int32Type = PassAuxiliary->Int32Type;

      std::vector<Value *> NewArgs;
      for(size_t ArgIndex = 0; ArgIndex < FunctionCall->getNumOperands(); ArgIndex++){
        NewArgs.push_back(FunctionCall->getArgOperand(ArgIndex));
      }


      errs() << *PassAuxiliary->CudaConfigureCall << "\n";





    return false;
    };
    

  
};