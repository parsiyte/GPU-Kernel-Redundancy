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
  
    SKE(char MemoryType, char Dimension, char Based) {
      this->MemoryType = tolower(MemoryType);
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

    char Dimension, Based, MemoryType; int TypeIndex;


    bool executeThePass(CallInst* FunctionCall, Auxiliary* PassAuxiliary) const override{

    IRBuilder<> Builder(FunctionCall); 
    BasicBlock* CurrentBB = FunctionCall->getParent();
    BasicBlock* PrevBB = CurrentBB->getPrevNode();
    BasicBlock* CheckPoint = nullptr;
    BasicBlock* ErrorBB  = nullptr;
    LLVMContext& Context = CurrentBB->getContext();

    std::pair<Instruction*, Instruction*> Errors;

    int RedundancyNumber = this->MemoryType == 'h' ? 2 : 1;
    BasicBlock* RedundantBBs[RedundancyNumber];
    Function* FunctionToReplicate = FunctionCall->getCalledFunction();
    Module* Module = FunctionToReplicate->getParent();
    std::string FunctionName = FunctionToReplicate->getName();
    
    std::vector<Value *> OrijinalArgs;
    std::vector<Value *> NewArgs;
    std::vector<Value *> CorrectionArgs;
    size_t NumberOfArg = FunctionCall->arg_size();
    for(size_t ArgIndex = 0; ArgIndex < NumberOfArg; ArgIndex++){
      Value* Arg = FunctionCall->getArgOperand(ArgIndex);
      NewArgs.push_back(Arg);
      if(LoadInst* ArgAsAlloca = dyn_cast<LoadInst>(Arg))
        Arg = ArgAsAlloca->getOperand(0);
      OrijinalArgs.push_back(Arg);
    }
    
    NewArgs.push_back(PassAuxiliary->CudaConfiguration[this->TypeIndex]);

    Output* OutputObject = PassAuxiliary->OutputObject;

    BasicBlock* SplitedNewBasicBlock = SplitBlock(PrevBB, PassAuxiliary->CudaConfigureCall);
    for(int Index = 0; Index < RedundancyNumber; Index++){
      RedundantBBs[Index] = createAndAllocateVariableAndreMemCpy(Builder, OutputObject, SplitedNewBasicBlock, PassAuxiliary, Index);
    }
    
    
    if(this->MemoryType == 'l'){
      CheckPoint = createCheckPoint(Builder, OutputObject, RedundantBBs[0], PassAuxiliary);
      ErrorBB = createErrorValue(Builder, CheckPoint, PassAuxiliary, Errors);
    }

    Instruction& LastInstructionOfPrevBB = PrevBB->back();
    LastInstructionOfPrevBB.setSuccessor(0, (ErrorBB == nullptr) ?  RedundantBBs[0]: ErrorBB);

    Instruction& LastInstructionOfLastRedundancyBB = RedundantBBs[RedundancyNumber-1]->back();
    LastInstructionOfLastRedundancyBB.setSuccessor(0, SplitedNewBasicBlock);

    for(int Index = RedundancyNumber -1 ; Index > 0; Index--){
      Instruction& LastInstructionOfRedundancyBB = RedundantBBs[Index-1]->back();
      LastInstructionOfRedundancyBB.setSuccessor(0, RedundantBBs[Index]);
    }

    Builder.SetInsertPoint(FunctionCall);
    Instruction** Replications =  OutputObject->Replications;
    CorrectionArgs.push_back(OutputObject->OutputAllocation);
    for(int ReplicationIndex = 0; ReplicationIndex < RedundancyNumber; ReplicationIndex++){
      Instruction* Replication = Replications[ReplicationIndex];
      Value* LoadedNewArg = Builder.CreateLoad(Replication);
      NewArgs.push_back(LoadedNewArg);
      CorrectionArgs.push_back(Replications[ReplicationIndex]);
    }
    

    FunctionType* NewKernelFunctionType = getTheNewKernelFunctionType(NewArgs, FunctionToReplicate->getReturnType());
    std::string NewKernelFunctionName = FunctionName + RevisitedSuffix + this->MemoryType +this->Based + this->Dimension;
    FunctionCallee NewKernelAsCallee =  Module->getOrInsertFunction(NewKernelFunctionName, NewKernelFunctionType);
    CallInst* NewFunctionCall = Builder.CreateCall(NewKernelAsCallee, NewArgs);
    Function* NewKernelFunction = cast<Function>(NewKernelAsCallee.getCallee());

    createHostRevisited(NewKernelFunction, FunctionToReplicate, PassAuxiliary, RedundancyNumber);
    
    FunctionCall->eraseFromParent();

    BasicBlock* NextBB = CurrentBB->getNextNode();
    Instruction* FirstInstructionOfNextBB = NextBB->getFirstNonPHI();
    Instruction* ClonedConfigure = PassAuxiliary->CudaConfigureCall->clone();
    ClonedConfigure->insertAfter(FirstInstructionOfNextBB);
    Builder.SetInsertPoint(ClonedConfigure->getNextNode());
    Instruction* ConfigureCheck = dyn_cast<Instruction>(Builder.CreateICmpNE(ClonedConfigure, PassAuxiliary->One32Bit));
    FirstInstructionOfNextBB = SplitBlockAndInsertIfThen(ConfigureCheck, ConfigureCheck->getNextNode(), false);
    Builder.SetInsertPoint(FirstInstructionOfNextBB);

    for(int Index = 0; Index < RedundancyNumber + 1; Index++){
      CorrectionArgs.at(Index) = Builder.CreateLoad(CorrectionArgs.at(Index));
    }

    CorrectionArgs.push_back(OutputObject->NumberOfElement);
    Function* CorrectionFunction;
    if(this->MemoryType == 'h')
     CorrectionFunction = OutputObject->MajorityVotingFunction;
    else{
      CorrectionFunction = OutputObject->DetectionFunction;
      CorrectionArgs.push_back(Builder.CreateLoad(Errors.second));
    } 
    
    Instruction* CorrectionCall = Builder.CreateCall(CorrectionFunction, CorrectionArgs);
    BasicBlock* CorrectionBB = CorrectionCall->getParent();
    BasicBlock* FinalBB = CorrectionBB->getNextNode();
    BasicBlock* ReplicationMemoryFree = FinalBB;
    for(int Index = 0; Index < RedundancyNumber; Index++)
      ReplicationMemoryFree = CudaMemoryFree(Builder, ReplicationMemoryFree, OutputObject, Index, PassAuxiliary);

    Instruction& LastInstructionOfCorrentionBlock = CorrectionBB->back();
    LastInstructionOfCorrentionBlock.setSuccessor(0, ReplicationMemoryFree);


    if(this->MemoryType == 'l'){
      BasicBlock *FirstErrorThen, *SecondErrorThen; 
      BasicBlock* ErrorCheckBB = createErrorCheck(Builder, FinalBB, PassAuxiliary, Errors, &FirstErrorThen);
      Instruction& ErrorCheckBranch = ErrorCheckBB->back();
      BasicBlock* NoError = ErrorCheckBranch.getSuccessor(1);


      Instruction& LastInstructionOfMemoryFree = ReplicationMemoryFree->back();
      LastInstructionOfMemoryFree.setOperand(0, ErrorCheckBB);

      ResetTheErroValues(Builder, PassAuxiliary, Errors, FirstErrorThen);
      BasicBlock* CheckpointBB = createNewOutputFromCheckPoint(Builder, OutputObject, PassAuxiliary, NoError, 1);
      Instruction& LastInstructionOfFirstErrorThen = FirstErrorThen->back();
      LastInstructionOfFirstErrorThen.setOperand(0, CheckpointBB);

      OrijinalArgs.back() = OutputObject->OutputAllocation;
      std::pair<Value*, Value*> NewDimensions;
      BasicBlock* SecondFunctionCallBB = createRedundantCall(Builder, FunctionToReplicate, OrijinalArgs, NoError, PassAuxiliary);
      BasicBlock *NewDimension = createNewDimension(Builder, SecondFunctionCallBB, PassAuxiliary, this->TypeIndex, NewDimensions);
      BasicBlock* SecondConfiguration = createConfiguration(Builder, SecondFunctionCallBB, NextBB, PassAuxiliary, false, NewDimensions, nullptr, 0 , this->TypeIndex);
      
      Instruction& LastInstructionOfCheckPointBB= CheckpointBB->back();
      LastInstructionOfCheckPointBB.setOperand(0, NewDimension);

      Instruction& LastInstructionOfNewDimesionBB= NewDimension->back();
      LastInstructionOfNewDimesionBB.setOperand(0,SecondConfiguration );
      //BasicBlock* SecondDetectionBB = createDetectionFunctionCall(Builder, FinalBB, OutputObject, 1, (Errors).second);
      //BasicBlock* SecondDetectionConfiguration = createConfiguration(Builder, SecondDetectionBB, FinalBB, PassAuxiliary);  
      Instruction& LastInstructionOfSecondFunctionCallBB = SecondFunctionCallBB->back();
      //LastInstructionOfSecondFunctionCallBB.setOperand(0, SecondDetectionConfiguration);

      //BasicBlock* SecondErrorCheckBB = createErrorCheck(Builder, FinalBB, PassAuxiliary, Errors, &SecondErrorThen);
      //Instruction& LastInstructionOfSecondDetectionCallBB = SecondDetectionBB->back();
      //LastInstructionOfSecondDetectionCallBB.setOperand(0, SecondErrorCheckBB);
      //RestoreFromTheErrorAndFreeRedundantOutput(Builder, OutputObject, PassAuxiliary, SecondErrorThen, 1);
      //Instruction& SecondErrorCheckBranch = SecondErrorCheckBB->back();
      //BasicBlock* SecondNoError = SecondErrorCheckBranch.getSuccessor(1);
      
      //Free(Builder, SecondNoError, OutputObject->Replications[1], PassAuxiliary, 1);
      Free(Builder, NoError, OutputObject->Checkpoint, PassAuxiliary, 0);
      Free(Builder, NoError, Errors.second, PassAuxiliary, 1);
      Free(Builder, NoError, Errors.first, PassAuxiliary, 0);
      //SecondNoError->back().setSuccessor(0, NoError);
      NoError->moveBefore(FinalBB);


    }


    Value* ConfigurationToMultiplicate = PassAuxiliary->CudaConfiguration[this->TypeIndex];
    CallInst* DimensionCall = dyn_cast<CallInst>(PassAuxiliary->CudaConfiguration[this->TypeIndex+4]);
    Builder.SetInsertPoint(DimensionCall);
    Value* Multiplier = ConstantInt::get(PassAuxiliary->Int32Type, RedundancyNumber+1);
    if(!DimensionCall->hasMetadata("Multiplication")){
      DimensionCall->setArgOperand(this->TypeIndex%2+1, Builder.CreateMul(ConfigurationToMultiplicate, Multiplier));
      MDNode* N = MDNode::get(Context, MDString::get(Context, "true"));
      dyn_cast<Instruction>(DimensionCall)->setMetadata("Multiplication", N);
      errs() << *DimensionCall << "\n";
    }else{
      NewFunctionCall->setArgOperand(NewArgs.size() - (1 + RedundancyNumber), Builder.CreateUDiv(ConfigurationToMultiplicate, Multiplier));
    }
    
    return false;
    };
    

  
};