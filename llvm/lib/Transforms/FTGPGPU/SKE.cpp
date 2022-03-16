#include "llvm/ADT/StringRef.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/raw_ostream.h"
#include <cstddef>
#include <string>
#include "FTGPGPUPass.cpp"


using namespace llvm;


class SKE : public AbstractPass{
  
  public:
  
    SKE(char Dimension, char Based) {
      this->Dimension = tolower(Dimension);
      this->Based = tolower(Based);
      this->TypeIndex = calculateTypeIndex();
      }
    
    int calculateTypeIndex(){
      int DimnesionIndex = 0, BasedIndex = 0;
      if(this->Dimension == 'y')
        DimnesionIndex = 1;
      if(this->Based == 't')
        BasedIndex = 1;
      
    return 2 * BasedIndex + DimnesionIndex;
    }

    char Dimension, Based; int TypeIndex;


    bool executeThePass(CallInst* FunctionCall, Auxiliary* PassAuxiliary) const override{

    BasicBlock* CurrentBB = FunctionCall->getParent();
    BasicBlock* NextBB = CurrentBB->getNextNode();
    BasicBlock* PrevBB = CurrentBB->getPrevNode();
    LLVMContext& Context = CurrentBB->getContext();

    

    Instruction* FirstInstructionOfNextBB = NextBB->getFirstNonPHI();
    Instruction* FirstInstructionOfPrevBB = PrevBB->getFirstNonPHI();
    Instruction& LastInstructionOfPrevBB = PrevBB->back();

    Function* FunctionToReplicate = FunctionCall->getCalledFunction();
    std::string FunctionName = FunctionToReplicate->getName();


    Module* Module = FunctionToReplicate->getParent();

    std::vector<Value *> NewArgs;
    size_t NumberOfArg = FunctionCall->arg_size();
    for(size_t ArgIndex = 0; ArgIndex < NumberOfArg; ArgIndex++){
      NewArgs.push_back(FunctionCall->getArgOperand(ArgIndex));
    }
    
    IRBuilder<> Builder(FunctionCall); 
    bool IsLoop = isForLoop(PrevBB);
    Output* OutputObject = PassAuxiliary->OutputObject;
    
    
    createAndAllocateVariableAndreMemCpy(Builder, OutputObject, LastInstructionOfPrevBB, PassAuxiliary->CudaMemCopy, IsLoop);


    Builder.SetInsertPoint(FunctionCall);
    Instruction** Replications =  OutputObject->Replications;
    for(int ReplicationIndex = 0; ReplicationIndex < NumberOfReplication - 1; ReplicationIndex++){
      Instruction* Replication = Replications[ReplicationIndex];
      Value* LoadedNewArg = Builder.CreateLoad(Replication);
      NewArgs.push_back(LoadedNewArg);
    }
    errs() << *PassAuxiliary->CudaConfiguration[this->TypeIndex] << "BURASI\n";
    NewArgs.push_back(PassAuxiliary->CudaConfiguration[this->TypeIndex]);

    FunctionType* NewKernelFunctionType = getTheNewKernelFunctionType(NewArgs, FunctionToReplicate->getReturnType());

    std::string NewKernelFunctionName = FunctionName + RevisitedSuffix + this->Based + this->Dimension;
    
    FunctionCallee NewKernelAsCallee =  Module->getOrInsertFunction(NewKernelFunctionName, NewKernelFunctionType);
    CallInst* NewFunctionCall = Builder.CreateCall(NewKernelAsCallee, NewArgs);
    Function* NewKernelFunction = cast<Function>(NewKernelAsCallee.getCallee());


    createHostRevisited(NewKernelFunction, FunctionToReplicate, PassAuxiliary);

    FunctionCall->eraseFromParent();  

    Instruction* ClonedConfigure = PassAuxiliary->CudaConfigureCall->clone();
    ClonedConfigure->insertAfter(FirstInstructionOfNextBB);
    Builder.SetInsertPoint(ClonedConfigure->getNextNode());
    Instruction* ConfigureCheck = dyn_cast<Instruction>(Builder.CreateICmpNE(ClonedConfigure, PassAuxiliary->One32Bit));
    FirstInstructionOfNextBB = SplitBlockAndInsertIfThen(ConfigureCheck, ConfigureCheck->getNextNode(), false);
    Builder.SetInsertPoint(FirstInstructionOfNextBB);

    
    Value* OrijinalOutput = OutputObject->OutputAllocation;
    Value* FirstReplicationOutput = OutputObject->Replications[0];
    Value* SecondReplicationOutput = OutputObject->Replications[1];
    Value* SizeOfOutput = OutputObject->MallocInstruction->getArgOperand(1);

    Value* LoadedOrijinalOutput = Builder.CreateLoad(OrijinalOutput);
    Value* LoadedFirstReplicationOutput = Builder.CreateLoad(FirstReplicationOutput);
    Value* LoadedSecondReplicationOutput = Builder.CreateLoad(SecondReplicationOutput);

    
    Builder.CreateCall(OutputObject->MajorityVotingFunction, {LoadedOrijinalOutput, LoadedFirstReplicationOutput, LoadedSecondReplicationOutput, SizeOfOutput});

    Value* ConfigurationToMultiplicate = PassAuxiliary->CudaConfiguration[this->TypeIndex];
    CallInst* DimensionCall = dyn_cast<CallInst>(PassAuxiliary->CudaConfiguration[this->TypeIndex+4]);
    Builder.SetInsertPoint(DimensionCall);
    if(!DimensionCall->hasMetadata("Multiplication")){
      DimensionCall->setArgOperand(this->TypeIndex%2+1, Builder.CreateMul(ConfigurationToMultiplicate, PassAuxiliary->Three32Bit));
      MDNode* N = MDNode::get(Context, MDString::get(Context, "true"));
      dyn_cast<Instruction>(DimensionCall)->setMetadata("Multiplication", N);
    }else{
      NewFunctionCall->setArgOperand(NewArgs.size() - 1, Builder.CreateUDiv(ConfigurationToMultiplicate, PassAuxiliary->Three32Bit));
    }

    return false;
    };
    

  
};