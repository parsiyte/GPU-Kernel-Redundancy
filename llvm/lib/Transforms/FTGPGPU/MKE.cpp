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


class MKE : public AbstractPass{
  
  public:
  
    MKE(char MemoryType, bool StreamEnabled = false) { this->MemoryType = MemoryType; this->StreamEnabled = StreamEnabled;}
    bool StreamEnabled;
    char MemoryType;
    
    bool executeThePass(CallInst* FunctionCall, Auxiliary* PassAuxiliary) const override{
      BasicBlock* CurrentBB = FunctionCall->getParent();
      BasicBlock* NextBB = CurrentBB->getNextNode();
      BasicBlock* PrevBB = CurrentBB->getPrevNode();
      Value* StreamArray = nullptr;
      Function* FunctionToReplicate = FunctionCall->getCalledFunction();
      FunctionCallee StreamCreateFunction = PassAuxiliary->StreamCreateFunction;
      int NumberOfRedundacy = this->MemoryType == 'H' ? 2 : 1;

      Output* OutputObject = PassAuxiliary->OutputObject;
      bool IsLoop = isForLoop(PrevBB);

  
      Instruction* FirstInstructionOfNextBB = NextBB->getFirstNonPHI();
      Instruction* FirstInstructionOfPrevBB = PrevBB->getFirstNonPHI();
     
      IRBuilder<> Builder(FunctionCall); 
      Value* Zero32bit = PassAuxiliary->Zero32Bit;
      Value* One32bit = PassAuxiliary->One32Bit;
      Type* Int32Type = PassAuxiliary->Int32Type;
      Value* CudaStreamNonBlocking = PassAuxiliary->CudaStreamNonBlocking;
      BasicBlock* NewBasicBlock = nullptr;
      if(this->StreamEnabled == true){
        ArrayType *ArrayType = ArrayType::get(PassAuxiliary->StreamType, NumberOfReplication);
        Builder.SetInsertPoint(FirstInstructionOfPrevBB);
        StreamArray = Builder.CreateAlloca(ArrayType, nullptr, "streams");
        Value* IthStream = Builder.CreateInBoundsGEP(StreamArray, {Zero32bit, Zero32bit}, "arrayidx"); // Bunu zaten özgün çağrıya verdiğimiz için direkt 0 verebiliriz.
        Builder.CreateCall(StreamCreateFunction, {IthStream, CudaStreamNonBlocking});
        Value* LoadedStream = Builder.CreateLoad(IthStream);
        PassAuxiliary->CudaConfigureCall->setArgOperand(StreamArgIndex, LoadedStream);

      }
      std::vector<Value *> ArgsOfReplicationFunction;
      for (Value* Arg : FunctionCall->args()) {
        Value* ArgToReplicate = Arg;
        if(LoadInst* ArgAsInstruction = dyn_cast_or_null<LoadInst>(ArgToReplicate))
          ArgToReplicate = ArgAsInstruction->getPointerOperand();
        ArgsOfReplicationFunction.push_back(ArgToReplicate);
      }

    
      if(this->MemoryType == 'H'){
        std::pair<Value*, Value*> NewDimensions;  
        BasicBlock* SplitedNewBasicBlock = SplitBlock(PrevBB, PassAuxiliary->CudaConfigureCall);
        Builder.SetInsertPoint(PassAuxiliary->CudaConfigureCall);
        Builder.CreateCall(PassAuxiliary->CudaThreadSync);
        BasicBlock* Redundant = createAndAllocateVariableAndreMemCpy(Builder, OutputObject, SplitedNewBasicBlock, PassAuxiliary, 0);
        BasicBlock* SecondRedundant = createAndAllocateVariableAndreMemCpy(Builder, OutputObject, Redundant, PassAuxiliary, 1);
        ArgsOfReplicationFunction.back() = OutputObject->Replications[0];
        BasicBlock* FunctionCallBB = createRedundantCall(Builder, FunctionToReplicate, ArgsOfReplicationFunction, NextBB, PassAuxiliary);
        BasicBlock* Configuration = createConfiguration(Builder, FunctionCallBB, NextBB, PassAuxiliary, false, NewDimensions, StreamArray, 1);
        Instruction& LastInstructionOfCurrentBB = CurrentBB->back();
        LastInstructionOfCurrentBB.setOperand(0, Configuration);
        Instruction& LastInstructionOfPrevBB = PrevBB->back();
        LastInstructionOfPrevBB.setOperand(0, SecondRedundant);


        ArgsOfReplicationFunction.back() = OutputObject->Replications[1];
        BasicBlock* SecondFunctionCallBB = createRedundantCall(Builder, FunctionToReplicate, ArgsOfReplicationFunction, NextBB, PassAuxiliary);
        BasicBlock* SecondConfiguration = createConfiguration(Builder, SecondFunctionCallBB, NextBB, PassAuxiliary, false, NewDimensions, StreamArray, 2);

        
        Instruction& LastInstructionOfRedundantBB = FunctionCallBB->back();
        LastInstructionOfRedundantBB.setOperand(0, SecondConfiguration);


        BasicBlock* majorityVotingBB = createMajorityVotingFunctionCall(Builder, NextBB, OutputObject);
        BasicBlock* majorityVotingConfiguration = createConfiguration(Builder, majorityVotingBB, NextBB, PassAuxiliary, this->StreamEnabled, NewDimensions);


        Instruction& LastInstructionOfSecondRedundantFunctionBB = SecondFunctionCallBB->back();
        LastInstructionOfSecondRedundantFunctionBB.setOperand(0, majorityVotingConfiguration);

        
        Free(Builder, NextBB, OutputObject->Replications[0], PassAuxiliary, 1);
        Free(Builder, NextBB, OutputObject->Replications[1], PassAuxiliary, 1);
        
        

      }else{
        std::pair<Instruction*, Instruction*> Errors;
        BasicBlock* SplitedNewBasicBlock = SplitBlock(PrevBB, PassAuxiliary->CudaConfigureCall);
        BasicBlock* Redundant = createAndAllocateVariableAndreMemCpy(Builder, OutputObject, SplitedNewBasicBlock, PassAuxiliary, 0);
        std::pair<Value*, Value*> NewDimensions;
        BasicBlock* CheckPoint = createCheckPoint(Builder, OutputObject, Redundant, PassAuxiliary);
        BasicBlock* ErrorBB = createErrorValue(Builder, CheckPoint, PassAuxiliary, Errors);
        ArgsOfReplicationFunction.back() = OutputObject->Replications[0];
        BasicBlock* FunctionCallBB = createRedundantCall(Builder, FunctionToReplicate, ArgsOfReplicationFunction, NextBB, PassAuxiliary);
        BasicBlock* Configuration = createConfiguration(Builder, FunctionCallBB, NextBB, PassAuxiliary, false, NewDimensions, StreamArray);
        Instruction& LastInstructionOfCurrentBB = CurrentBB->back();
        LastInstructionOfCurrentBB.setOperand(0, Configuration);
        Instruction& LastInstructionOfPrevBB = PrevBB->back();
        LastInstructionOfPrevBB.setOperand(0, ErrorBB);
        BasicBlock* DetectionBB = createDetectionFunctionCall(Builder, NextBB, OutputObject, 0, (Errors).second);
        BasicBlock* DetectionConfiguration = createConfiguration(Builder, DetectionBB, NextBB, PassAuxiliary, false, NewDimensions);
        Instruction& LastInstructionOfFunctionCallBB = FunctionCallBB->back();
        LastInstructionOfFunctionCallBB.setOperand(0, DetectionConfiguration);
        BasicBlock* FirstReplicationMemoryFree = CudaMemoryFree(Builder, NextBB, OutputObject, 0, PassAuxiliary);
        Instruction& LastInstructionOfDetectionCallBB = DetectionBB->back();
        LastInstructionOfDetectionCallBB.setOperand(0, FirstReplicationMemoryFree);
        BasicBlock *FirstErrorThen, *SecondErrorThen; 
        BasicBlock* ErrorCheckBB = createErrorCheck(Builder, NextBB, PassAuxiliary, Errors, &FirstErrorThen);
        Instruction& ErrorCheckBranch = ErrorCheckBB->back();
        BasicBlock* NoError = ErrorCheckBranch.getSuccessor(1);
        Instruction& LastInstructionOfMemoryFree = FirstReplicationMemoryFree->back();
        LastInstructionOfMemoryFree.setOperand(0, ErrorCheckBB);
        ResetTheErroValues(Builder, PassAuxiliary, Errors, FirstErrorThen);
        BasicBlock* CheckpointBB = createNewOutputFromCheckPoint(Builder, OutputObject, PassAuxiliary, NextBB, 1);
        Instruction& LastInstructionOfFirstErrorThen = FirstErrorThen->back();
        LastInstructionOfFirstErrorThen.setOperand(0, CheckpointBB);
        ArgsOfReplicationFunction.back() = OutputObject->OutputAllocation;
        BasicBlock* SecondFunctionCallBB = createRedundantCall(Builder, FunctionToReplicate, ArgsOfReplicationFunction, NextBB, PassAuxiliary);
        BasicBlock* SecondConfiguration = createConfiguration(Builder, SecondFunctionCallBB, NextBB, PassAuxiliary, true, NewDimensions);
        Instruction& LastInstructionOfCheckpointBB = CheckpointBB->back();
        LastInstructionOfCheckpointBB.setOperand(0, SecondConfiguration);
        Instruction& LastInstructionOfSecondFunctionCallBB = SecondFunctionCallBB->back();
        Free(Builder, NoError, OutputObject->Checkpoint, PassAuxiliary, 0);
        Free(Builder, NoError, Errors.second, PassAuxiliary, 1);
        Free(Builder, NoError, Errors.first, PassAuxiliary, 0);
        SecondFunctionCallBB->back().setSuccessor(0, NoError);
        NoError->moveBefore(NextBB);
      }



    return true;
    };
    

  
};