#ifndef MYHEADEFILE_H
#define MYHEADEFILE_H

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Argument.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/CallingConv.h"
#include "llvm/IR/Constant.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Use.h"
#include "llvm/IR/Value.h"
#include "llvm/Pass.h"
#include "llvm/Support/Alignment.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ConvertUTF.h"
#include "llvm/Support/raw_ostream.h"

#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"

#include "llvm-c/Initialization.h"
#include "llvm-c/Transforms/AggressiveInstCombine.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/BasicAliasAnalysis.h"
#include "llvm/Analysis/GlobalsModRef.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/IR/Type.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Transforms/AggressiveInstCombine/AggressiveInstCombine.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/Local.h"
#include <algorithm>
#include <cstddef>
#include <string>
#include <utility>
#include <vector>
using namespace llvm;


#define RedundantString "Redundancy"
#define NumberOfReplication 3
#define StreamCreateFunctionName "cudaStreamCreateWithFlags"
#define StreamArgIndex 5
#define RevisitedSuffix "Revisited"


struct Output {
  Function* MajorityVotingFunction;
  Function* DetectionFunction;
  AllocaInst* OutputAllocation;
  CallInst* MallocInstruction;
  Type* OutputType;
  Type* DestinationType;
  StringRef Name;
  Instruction* Replications[NumberOfReplication-1];
  Value* Checkpoint;
  Value* Size;
  Value* NumberOfElement;
};
/*
enum Dimensions { X, Y };
enum ReplicationBased { Block, Thread };

struct CudaConfigurations{
  Instruction* BlockX;
  Instruction* BlockY;
  Instruction* ThreadX;
  Instruction* ThreadY;
};
*/


struct Auxiliary{
  FunctionCallee StreamCreateFunction;
  FunctionCallee CudaMemCopy;
  FunctionCallee CudaThreadSync;
  FunctionCallee LogFunction;
  FunctionCallee FreeFunction;

  Function*  CudaGlobalRegisterFunction;
  Function*  CudaRegisterFunction;
  Function*  CudaSetupArgument;
  Function*  CudaLaunch;
  Function*  CudaMemFree;
  Function*  MallocFunction;
  Function*  CudaMalloc;

/*
  Function*  ThreadIDX;
  Function*  BlockIDX;
  Function*  BlockDimX;
*/
  FunctionCallee CudaDimensionFunctions[6];

  CallInst* CudaMallocFunction;
  CallInst* CudaMemCpyFunction;
  CallInst* CudaConfigureCall;

  Type* StreamType;
  Type* StreamPointType;
  Type* Int32Type;
  Type* Int64Type;
  Type* VoidType;

  PointerType* Int8PtrType;
  PointerType* Int32PtrType;
  ConstantPointerNull* Int8PtrNull;
  ConstantPointerNull* Int32PtrNull;

  Value* Zero32Bit;
  Value* Zero64Bit;
  Value* One32Bit;
  Value* Two32Bit;
  Value* Three32Bit;
  Value* MinusOne32Bit;
  Value* CudaStreamNonBlocking;
  Value* Twelwe64Bit;
  
  Output* OutputObject;

  Value* CudaConfiguration[8];


  Constant* Four64Bit;
  Constant* Eight64Bit;

};

inline void registerTheFunction(Function* FunctionToRegister, Auxiliary* PassAuxiliary){

  Function *CudaGlobalRegister = PassAuxiliary->CudaGlobalRegisterFunction;
  Function *CudaRegister = PassAuxiliary->CudaRegisterFunction;
  std::string FunctionName = FunctionToRegister->getName();
  BasicBlock *CudaRegisterBlock = dyn_cast<BasicBlock>(CudaGlobalRegister->begin());
  Instruction *FirstInstruction = dyn_cast<Instruction>(CudaRegisterBlock->begin());
  IRBuilder<> Builder(FirstInstruction);

  Value *FunctionNameAsGlobal =  Builder.CreateGlobalStringPtr(FunctionName);
  Builder.CreateCall( CudaRegister,
      {FirstInstruction->getOperand(0),
        Builder.CreateBitCast(FunctionToRegister, PassAuxiliary->Int8PtrType),
        FunctionNameAsGlobal, FunctionNameAsGlobal,
        PassAuxiliary->MinusOne32Bit,
        PassAuxiliary->Int8PtrNull,
        PassAuxiliary->Int8PtrNull,
        PassAuxiliary->Int8PtrNull,
        PassAuxiliary->Int8PtrNull,
        PassAuxiliary->Int32PtrNull});

}

inline Function *createDetection(Module& M, PointerType *DetectionPointerType, Auxiliary* PassAuxiliary, std::string DetectionFunctionName) {

  Function *CudaSetupArgument = PassAuxiliary->CudaSetupArgument;
  Function *CudaLaunch = PassAuxiliary->CudaLaunch;

  Type *Int64Type = PassAuxiliary->Int64Type;
  Type *Int32Type = PassAuxiliary->Int32Type;
  Type *Int32PtrType = PassAuxiliary->Int32PtrType;
  Value *Zero32bit = PassAuxiliary->Zero32Bit;

  PointerType *Int8PtrType = PassAuxiliary->Int8PtrType;

  FunctionCallee DetectionCallee = M.getOrInsertFunction(DetectionFunctionName, PassAuxiliary->VoidType, DetectionPointerType, DetectionPointerType, Int64Type, Int32PtrType);

  std::vector<Value *> Parameters;
  Function *DetectionFunction = dyn_cast<Function>(DetectionCallee.getCallee());

  DetectionFunction->setCallingConv(CallingConv::C);
  Function::arg_iterator Args = DetectionFunction->arg_begin();

  Value *A = Args++; A->setName("A");
  Value *B = Args++; B->setName("B");
  Value *Size = Args++; Size->setName("Size");
  Value *Error = Args++; Error->setName("Error");
  BasicBlock *EntryBlock = BasicBlock::Create(M.getContext(), "entry", DetectionFunction);
  IRBuilder<> Builder(EntryBlock); Builder.SetInsertPoint(EntryBlock);
  Value *Aptr = Builder.CreateAlloca(DetectionPointerType, nullptr, "A.addr");
  Value *Bptr = Builder.CreateAlloca(DetectionPointerType, nullptr, "B.addr");
  Value *Sizeptr = Builder.CreateAlloca(Int64Type, nullptr, "size.addr");
  Value *ErrorPtr = Builder.CreateAlloca(Int32PtrType, nullptr, "error.addr");

  Builder.CreateStore(A, Aptr);
  Builder.CreateStore(B, Bptr);
  Builder.CreateStore(Size, Sizeptr);
  Builder.CreateStore(Error, ErrorPtr);

  Parameters.push_back(Aptr);
  Parameters.push_back(Bptr);
  Parameters.push_back(Sizeptr);
  Parameters.push_back(ErrorPtr);

  int Offset = 0;
  int SizeParameter = 8;
  for (unsigned long Index = 0; Index < Parameters.size(); Index++) {
    Value *Parameter = Parameters.at(Index);
    Value *BitcastParameter = Builder.CreateBitCast(Parameter, Int8PtrType);
    Value *OffsetValue = ConstantInt::get(Int64Type, Offset);

    if(dyn_cast_or_null<PointerType>(Parameter->getType()) == nullptr) SizeParameter = 4;
    else SizeParameter = 8;

    Value *SizeValue = ConstantInt::get(Int64Type, SizeParameter);
    Value *CudaSetupArgumentCall = Builder.CreateCall( CudaSetupArgument, {BitcastParameter, SizeValue, OffsetValue});
    Instruction *IsError = dyn_cast<Instruction>( Builder.CreateICmpEQ(CudaSetupArgumentCall, Zero32bit));
    if (Index == 0) Builder.CreateRetVoid(); 

    Instruction *SplitPoint = SplitBlockAndInsertIfThen(IsError, IsError->getNextNode(), false);

    SplitPoint->getParent()->setName("setup.next");

    Builder.SetInsertPoint(SplitPoint);
    Offset += SizeParameter;
  }

  Builder.CreateCall(CudaLaunch, {Builder.CreateBitCast( DetectionFunction , Int8PtrType)});
  registerTheFunction(DetectionFunction, PassAuxiliary);

  return DetectionFunction;
}

inline void createOrInsertDetectionFunction(Module& M, Output* OutputObject, Auxiliary* PassAuxiliary){
  Type* OutputType = OutputObject->OutputType;
  std::string DetectionFunctionName = "Detection" + std::to_string(OutputType->getPointerElementType()->getTypeID()); 
  Function* DetectionFunction = M.getFunction(DetectionFunctionName);

  if(DetectionFunction == nullptr) DetectionFunction = createDetection(M, dyn_cast<PointerType>(OutputType), PassAuxiliary, DetectionFunctionName);
  OutputObject->DetectionFunction = DetectionFunction;
}


inline void* RestoreFromTheErrorAndFreeRedundantOutput(IRBuilder<> Builder, Output* Output, Auxiliary* PassAuxiliary, BasicBlock* ThenBlock, int ReplicationIndex){
  //LLVMContext& Context = InsertBefore->getContext();
  //BasicBlock* NewBasicBlock = BasicBlock::Create(Context, "RestoreFromTheError", InsertBefore->getParent(), InsertBefore);
  
  Builder.SetInsertPoint(ThenBlock->getFirstNonPHI());

  Instruction* LoadedOriginalOutput = Builder.CreateLoad(Output->OutputAllocation);
  Value* BitcastedOriginalOutput = Builder.CreateBitCast(LoadedOriginalOutput, PassAuxiliary->Int8PtrType);

  Instruction* LoadedRedundantOutput =  Builder.CreateLoad(Output->Replications[ReplicationIndex]);
  Value* BitcastedRedudantOutput = Builder.CreateBitCast(LoadedRedundantOutput, PassAuxiliary->Int8PtrType);
  
  
  Builder.CreateCall(PassAuxiliary->CudaMemCopy, {BitcastedOriginalOutput, BitcastedRedudantOutput, Output->Size, PassAuxiliary->Three32Bit});

  //FreeRedundantOutput(Builder, Output, PassAuxiliary, ReplicationIndex);

  //Builder.CreateBr(InsertBefore);
}

inline void Free(IRBuilder<> Builder, BasicBlock* InsertHere, Value* ToFree, Auxiliary* PassAuxiliary, int GPU){
  
  Builder.SetInsertPoint(InsertHere->getFirstNonPHI());
  FunctionCallee FreeFunction = GPU == 1 ? PassAuxiliary->CudaMemFree : PassAuxiliary->FreeFunction;
  Value* LoadedOutput = Builder.CreateLoad(ToFree);
  Value* BitcatededOutput = Builder.CreateBitCast(LoadedOutput, PassAuxiliary->Int8PtrType);

  Builder.CreateCall(FreeFunction, {BitcatededOutput});

}

inline BasicBlock* createNewOutputFromCheckPoint(IRBuilder<> Builder, Output* Output, Auxiliary* PassAuxiliary, BasicBlock* InsertBefore, int ReplicationIndex){
  LLVMContext& Context = InsertBefore->getContext();
  BasicBlock* NewBasicBlock = BasicBlock::Create(Context, "RestoreFromCheckpoint", InsertBefore->getParent(), InsertBefore);
  Builder.SetInsertPoint(NewBasicBlock);
  Instruction* RestoreOutput = Output->OutputAllocation;
  Value* NewOutputLoaded = Builder.CreateLoad(RestoreOutput);
  Value* BitcastedNewOutput = Builder.CreateBitCast(NewOutputLoaded, PassAuxiliary->Int8PtrType);
  Value* LoadedCPU = Builder.CreateLoad(Output->Checkpoint);
  Value* BitcastedCPU = Builder.CreateBitCast(LoadedCPU, PassAuxiliary->Int8PtrType);

  Builder.CreateCall(PassAuxiliary->CudaMemCopy, {BitcastedNewOutput, BitcastedCPU, Output->Size, PassAuxiliary->One32Bit});

  Builder.CreateBr(InsertBefore);

  return NewBasicBlock;
}

inline void* ResetTheErroValues(IRBuilder<> Builder, Auxiliary* PassAuxiliary, std::pair<Instruction*, Instruction* > &Errors, BasicBlock* ThenBlock){
  Builder.SetInsertPoint(ThenBlock->getFirstNonPHI());
  Value* LoadedErrorCPU = Builder.CreateLoad(Errors.first);
  Value* ErrorValuePtr = Builder.CreateGEP(LoadedErrorCPU, PassAuxiliary->Zero32Bit);
  Builder.CreateStore(PassAuxiliary->Zero32Bit, ErrorValuePtr);
  
  LoadedErrorCPU = Builder.CreateLoad(Errors.first); //ErrorCPU
  Value* BitcastedErrorCPU = Builder.CreateBitCast(LoadedErrorCPU, PassAuxiliary->Int8PtrType);

  Value* LoadedErrorGPU = Builder.CreateLoad(Errors.second); //ErrorGPU
  Value* BitcastedErrorGPU = Builder.CreateBitCast(LoadedErrorGPU, PassAuxiliary->Int8PtrType);

  Builder.CreateCall(PassAuxiliary->CudaMemCopy, {BitcastedErrorGPU, BitcastedErrorCPU, PassAuxiliary->Four64Bit, PassAuxiliary->One32Bit});

  ThenBlock->setName("ResetTheErrors");

}

inline BasicBlock* createErrorCheck(IRBuilder<> Builder, BasicBlock* InsertBefore, Auxiliary* PassAuxiliary, std::pair<Instruction*, Instruction* > &Errors, 
  BasicBlock **ThenBlock){
  LLVMContext& Context = InsertBefore->getContext();
  BasicBlock* NewBasicBlock = BasicBlock::Create(Context,"ErrorCheck", InsertBefore->getParent(), InsertBefore);
  Builder.SetInsertPoint(NewBasicBlock);

  Value* LoadedErrorCPU = Builder.CreateLoad(Errors.first); //ErrorCPU
  Value* BitcastedErrorCPU = Builder.CreateBitCast(LoadedErrorCPU, PassAuxiliary->Int8PtrType);


  Value* LoadedErrorGPU = Builder.CreateLoad(Errors.second); //ErrorGPU
  Value* BitcastedErrorGPU = Builder.CreateBitCast(LoadedErrorGPU, PassAuxiliary->Int8PtrType);

  Builder.CreateCall(PassAuxiliary->CudaMemCopy, {BitcastedErrorCPU, BitcastedErrorGPU, PassAuxiliary->Four64Bit, PassAuxiliary->Two32Bit});

  LoadedErrorCPU = Builder.CreateLoad(Errors.first);

  Value* ErrorValuePtr = Builder.CreateGEP(LoadedErrorCPU, PassAuxiliary->Zero32Bit);

  Value* ErrorValue = Builder.CreateLoad(ErrorValuePtr);

  Value* ErrorCheck = Builder.CreateICmpEQ(ErrorValue, PassAuxiliary->One32Bit);

  Instruction* Branch = Builder.CreateBr(InsertBefore);
  Instruction* ThenBlockBranch = SplitBlockAndInsertIfThen(ErrorCheck, Branch, false);

  BasicBlock* ThenBB = ThenBlockBranch->getParent();
  *ThenBlock = ThenBB;


  return NewBasicBlock;
}

inline BasicBlock* CudaMemoryFree(IRBuilder<> Builder, BasicBlock* InsertBefore, Output* Output, int ReplicationIndex, Auxiliary* PassAuxiliary){
  LLVMContext& Context = InsertBefore->getContext();
  BasicBlock* NewBasicBlock = BasicBlock::Create(Context, "CudaMemFree", InsertBefore->getParent(), InsertBefore);
  Builder.SetInsertPoint(NewBasicBlock);
  std::vector<Value *> MemFreeArgs;
  MemFreeArgs.push_back(Builder.CreateBitCast(Builder.CreateLoad(Output->Replications[ReplicationIndex]), PassAuxiliary->Int8PtrType));
  

  Builder.CreateCall(PassAuxiliary->CudaMemFree, MemFreeArgs);
  Builder.CreateBr(InsertBefore);
  
  
  return NewBasicBlock;
}

inline BasicBlock* createMajorityVotingFunctionCall(IRBuilder<> Builder, BasicBlock* InsertBefore, Output* Output){
  LLVMContext& Context = InsertBefore->getContext();
  BasicBlock* NewBasicBlock = BasicBlock::Create(Context, "MajorityVoting", InsertBefore->getParent(), InsertBefore);
  Builder.SetInsertPoint(NewBasicBlock);
  std::vector<Value *> MajorityVotingCallArgs;
  MajorityVotingCallArgs.push_back(Builder.CreateLoad(Output->OutputAllocation));
  MajorityVotingCallArgs.push_back(Builder.CreateLoad(Output->Replications[0]));
  MajorityVotingCallArgs.push_back(Builder.CreateLoad(Output->Replications[1]));
  MajorityVotingCallArgs.push_back(Output->Size);
  Builder.CreateCall(Output->MajorityVotingFunction, MajorityVotingCallArgs);
  Builder.CreateBr(InsertBefore);
  return NewBasicBlock;
}


inline BasicBlock* createDetectionFunctionCall(IRBuilder<> Builder, BasicBlock* InsertBefore, Output* Output, int ReplicationIndex, Value* GPUError){
  LLVMContext& Context = InsertBefore->getContext();
  BasicBlock* NewBasicBlock = BasicBlock::Create(Context, "DetectionCall", InsertBefore->getParent(), InsertBefore);
  Builder.SetInsertPoint(NewBasicBlock);
  std::vector<Value *> DetectionCallArgs;
  DetectionCallArgs.push_back(Builder.CreateLoad(Output->OutputAllocation));
  DetectionCallArgs.push_back(Builder.CreateLoad(Output->Replications[ReplicationIndex]));
  DetectionCallArgs.push_back(Output->Size);
  DetectionCallArgs.push_back(Builder.CreateLoad(GPUError));
  Builder.CreateCall(Output->DetectionFunction, DetectionCallArgs);
  Builder.CreateBr(InsertBefore);
  return NewBasicBlock;
}

inline BasicBlock* createNewDimension(IRBuilder<> Builder, BasicBlock* InsertBefore, Auxiliary* PassAuxiliary, int TypeIndex, std::pair<Value*, Value*> &NewDimensions){
  LLVMContext& Context = InsertBefore->getContext();
  StructType* StructTypeForTmp = StructType::create({PassAuxiliary->Int64Type, PassAuxiliary->Int32Type});
  BasicBlock* NewBasicBlock = BasicBlock::Create(Context, "NewDimension", InsertBefore->getParent());
  Builder.SetInsertPoint(NewBasicBlock);
  Instruction* DimesionCall = dyn_cast<CallInst>(PassAuxiliary->CudaConfiguration[TypeIndex+4])->clone();
  Value* GridOrBlock = DimesionCall->getOperand(0);
  AllocaInst* TempArg = Builder.CreateAlloca(dyn_cast<PointerType>(GridOrBlock->getType())->getElementType());
  AllocaInst* TempArgCoerce = Builder.CreateAlloca(StructTypeForTmp);
  DimesionCall->insertBefore(TempArg);
  Value* BitcastedTemp = Builder.CreateBitCast(TempArg, PassAuxiliary->Int8PtrType);
  Value* BitcastedGridOrBlock = Builder.CreateBitCast(GridOrBlock, PassAuxiliary->Int8PtrType);
  Builder.CreateMemCpy(BitcastedTemp, MaybeAlign(4), BitcastedGridOrBlock, MaybeAlign(4), PassAuxiliary->Twelwe64Bit);
  Value* BitcastedTempArgCoerce = Builder.CreateBitCast(TempArgCoerce, PassAuxiliary->Int8PtrType);
   BitcastedTemp = Builder.CreateBitCast(TempArg, PassAuxiliary->Int8PtrType);
  Builder.CreateMemCpy(BitcastedTempArgCoerce, MaybeAlign(4), BitcastedTemp, MaybeAlign(4), PassAuxiliary->Twelwe64Bit);
  Value* NthValueX = Builder.CreateInBoundsGEP(TempArgCoerce, {PassAuxiliary->Zero32Bit,  PassAuxiliary->Zero32Bit});
  Value* LoadedX = Builder.CreateLoad(NthValueX);
  Value* NthValueY = Builder.CreateInBoundsGEP(TempArgCoerce, {PassAuxiliary->Zero32Bit, PassAuxiliary->One32Bit });
  Value* LoadedY = Builder.CreateLoad(NthValueY);
  Builder.CreateBr(InsertBefore);
  //std::pair<Value*, Value*> tmpPair = std::make_pair(LoadedX, LoadedY);
  NewDimensions = std::make_pair(LoadedX, LoadedY);
  return NewBasicBlock;
}

inline BasicBlock* createConfiguration(IRBuilder<> Builder, BasicBlock* FunctionCallBB, BasicBlock* InsertBefore, 
Auxiliary* PassAuxiliary,bool isSync, std::pair<Value*, Value*> &NewDimensions, Value* StreamArray = nullptr, int StreamIndex = 0, int TypeIndex = -1){
  LLVMContext& Context = InsertBefore->getContext();
  BasicBlock* NewBasicBlock = BasicBlock::Create(Context, "CudaConfiguration", InsertBefore->getParent());
  Builder.SetInsertPoint(NewBasicBlock);
  if(isSync)
    Builder.CreateCall(PassAuxiliary->CudaThreadSync);
  Instruction* NewConfigurationCall = (PassAuxiliary->CudaConfigureCall)->clone();
  if(StreamArray != nullptr){
      Value* SecondStream = Builder.CreateInBoundsGEP(StreamArray, {PassAuxiliary->Zero32Bit, ConstantInt::get(PassAuxiliary->Int32Type, StreamIndex)}, "arrayidx");
      Builder.CreateCall(PassAuxiliary->StreamCreateFunction, {SecondStream, PassAuxiliary->CudaStreamNonBlocking});
      Value* LoadedStream = Builder.CreateLoad(SecondStream);
      NewConfigurationCall->setOperand(StreamArgIndex, LoadedStream);
  }
  if(TypeIndex != -1){
    int Index = TypeIndex/2;
    NewConfigurationCall->setOperand(2*Index, NewDimensions.first);
    NewConfigurationCall->setOperand(2*Index+1, NewDimensions.second);

  }
  Instruction* CMP = dyn_cast<Instruction>(Builder.CreateICmpNE(NewConfigurationCall, PassAuxiliary->One32Bit));
  NewConfigurationCall->insertBefore(CMP);
  Builder.CreateCondBr(CMP, FunctionCallBB, InsertBefore);
  NewBasicBlock->moveBefore(FunctionCallBB);
  return NewBasicBlock;
}

inline BasicBlock* createRedundantCall(IRBuilder<> Builder, Function* FunctionToReplicate, std::vector<Value *> ReplicationArgs, BasicBlock* InsertBefore, Auxiliary* PassAuxiliary){
  LLVMContext& Context = InsertBefore->getContext();
  BasicBlock* NewBasicBlock = BasicBlock::Create(Context, "RedundantFunctionCall", InsertBefore->getParent(), InsertBefore);
  Builder.SetInsertPoint(NewBasicBlock);
  std::vector<Value *> FunctionCallArgs;
  for (auto& Args : ReplicationArgs) {
    if(AllocaInst* Arg = dyn_cast<AllocaInst>(Args))
      FunctionCallArgs.push_back(Builder.CreateLoad(Args));
    else
      FunctionCallArgs.push_back(Args);

  }

  Builder.CreateCall(FunctionToReplicate, FunctionCallArgs);
  Builder.CreateBr(InsertBefore);


  return NewBasicBlock;
}


inline void changeXID(Function *Kernel, Value* NewCreatedCall, Value *ToBeChange, IRBuilder<> Builder) {
  for (Function::iterator BB = Kernel->begin(); BB != Kernel->end(); ++BB) {
    for (BasicBlock::iterator CurrentInstruction = BB->begin(); CurrentInstruction != BB->end(); ++CurrentInstruction) {
      if (CallInst *FunctionCall = dyn_cast<CallInst>(CurrentInstruction)) {
        if(FunctionCall == NewCreatedCall)
          continue;
        StringRef FunctionName = FunctionCall->getCalledFunction()->getName();
        if (FunctionName == "llvm.nvvm.read.ptx.sreg.ctaid.x") {
          FunctionCall->replaceAllUsesWith(Builder.CreateLoad(ToBeChange));
        } 
      }
    }
  }
}

inline Instruction* createNewID(IRBuilder<> Builder, Value* OriginalBasedaddr, Auxiliary* PassAuxiliary, int SchemeID) {
  int Based = SchemeID / 2;
  int Dimension = SchemeID % 2;
  Value* BlockId = Builder.CreateCall(PassAuxiliary->CudaDimensionFunctions[Dimension]);
  Value* NormalizedBlockID = Based == 0 ? Builder.CreateURem(BlockId, Builder.CreateLoad(OriginalBasedaddr)) : BlockId;
  Value* BlockDim = Based == 0 ? Builder.CreateCall(PassAuxiliary->CudaDimensionFunctions[4 + Dimension]) : dyn_cast<Value>(Builder.CreateLoad(OriginalBasedaddr));
  Value* BlockIdLocation = Builder.CreateMul(BlockDim, NormalizedBlockID);
  Value* ThreadID = Builder.CreateCall(PassAuxiliary->CudaDimensionFunctions[Dimension + 2]);
  Value* NormalizedThreadID = Based == 0 ? ThreadID: Builder.CreateURem(ThreadID, Builder.CreateLoad(OriginalBasedaddr));
  Value* newThreadId = Builder.CreateAdd(BlockIdLocation, NormalizedThreadID);
  return dyn_cast<Instruction>(newThreadId);
}

inline void changeTheID(Function* NewKernelFunction, Instruction* NewThreadID, StringRef FunctionName){
  Instruction* TempInstruction = NewThreadID;
  Instruction* PrevInst = TempInstruction;
  do{
    CallInst* TmpAsCall = dyn_cast_or_null<CallInst>(TempInstruction);
    if((TmpAsCall != nullptr && TmpAsCall->getCalledFunction()->getName() == FunctionName))
      break;
    PrevInst = TempInstruction;
    TempInstruction = TempInstruction->getNextNode();
    if(TempInstruction == nullptr){ 
      if(PrevInst->getParent()->getNextNode()){
        break;
      }
      TempInstruction = PrevInst->getParent()->getNextNode()->getFirstNonPHI();
    }
  } while (true);

  if(TempInstruction != nullptr){
    TempInstruction = TempInstruction->getNextNode();
    TempInstruction->replaceAllUsesWith(NewThreadID);
  }
}



inline void alterTheFunction(Function *NewKernelFunction, Auxiliary* PassAuxiliary, int SchemeID, int NumberofRedundant){
  Value* RedudantArgs[NumberofRedundant];
  Instruction* RedudantAllocation[NumberofRedundant];
  Function::arg_iterator Args = NewKernelFunction->arg_end();
  Args--;
  for(int Index = 0; Index < NumberofRedundant; Index++){
    RedudantArgs[Index] = Args--;
    RedudantArgs[Index]->setName("Redudant");
  }
  Value *OriginalBased = Args--;
  Value *OriginalOutput = Args--;

  Value* Output = NewKernelFunction->getArg(NewKernelFunction->arg_size() - 2 - NumberofRedundant);
  Type* OutputType = OriginalOutput->getType();

  User* FirstUser =  Output->uses().begin()->getUser();
  StoreInst* OutputStore = dyn_cast<StoreInst>(FirstUser);
  AllocaInst* OutputAllocation = dyn_cast<AllocaInst>(OutputStore->getPointerOperand());

  BasicBlock* FirstBB = dyn_cast<BasicBlock>(NewKernelFunction->begin());
  
  Instruction& LastInstruction = FirstBB->front();

  std::string ValueName = OriginalOutput->getName().str();
  OriginalBased->setName("OriginalBased");

  IRBuilder<> Builder(OutputAllocation->getNextNode());
  for(int Index = 0; Index < NumberofRedundant; Index++){
    RedudantAllocation[Index] =  Builder.CreateAlloca(OutputType, nullptr, "redundant.addr");
  }
  Value* OriginalBaseddr = Builder.CreateAlloca(PassAuxiliary->Int32Type,nullptr, "OriginalBased.addr");
  Instruction* MetaOutput = Builder.CreateAlloca(OutputType,nullptr, "MetaOutput");  

  Builder.CreateStore(OriginalBased, OriginalBaseddr);


  for(int Index = 0; Index < NumberofRedundant; Index++)
    Builder.CreateStore(RedudantArgs[Index], RedudantAllocation[Index]);

  OutputAllocation->replaceAllUsesWith(MetaOutput);
    

  Builder.CreateStore(Output, OutputAllocation);
    

  Instruction* RedundantIDAddr =  Builder.CreateAlloca(PassAuxiliary->Int32Type, nullptr, "RedundantIDAddr");
  Value* RedundantIDCall = Builder.CreateCall(PassAuxiliary->CudaDimensionFunctions[SchemeID]);

  Value* RedundantID = Builder.CreateUDiv(RedundantIDCall, Builder.CreateLoad(OriginalBaseddr));
  RedundantID->setName("RedundantID");
  Builder.CreateStore(RedundantID, RedundantIDAddr);


  Instruction* NewThreadID = createNewID(Builder,OriginalBaseddr, PassAuxiliary, SchemeID);
  changeTheID(NewKernelFunction, NewThreadID, PassAuxiliary->CudaDimensionFunctions[SchemeID%2+2].getCallee()->getName());


    
  Builder.SetInsertPoint(OutputStore->getNextNode());
  RedundantID = Builder.CreateLoad(RedundantIDAddr);
  Instruction* ZeroCmp = dyn_cast<Instruction>(Builder.CreateICmpEQ(RedundantID, PassAuxiliary->Zero32Bit));
  Instruction *ThenTerm, *FirstElseIfCondTerm;
  SplitBlockAndInsertIfThenElse(ZeroCmp, ZeroCmp->getNextNode(), &ThenTerm, &FirstElseIfCondTerm); 
  Builder.SetInsertPoint(ThenTerm);
  Builder.CreateStore(Builder.CreateLoad(OutputAllocation), MetaOutput);


  Instruction *ElseIfTerm, *SecondElseTerm;
  Builder.SetInsertPoint(FirstElseIfCondTerm);

  for(int Index = 0; Index < NumberofRedundant; Index++){
    RedundantID = Builder.CreateLoad(RedundantIDAddr);
    Instruction* CMP = dyn_cast<Instruction>(Builder.CreateICmpEQ(RedundantID, ConstantInt::get(PassAuxiliary->Int32Type, Index+1)));
    SplitBlockAndInsertIfThenElse(CMP, CMP->getNextNode(), &ElseIfTerm, &SecondElseTerm); 
    Builder.SetInsertPoint(ElseIfTerm);
    Builder.CreateStore(Builder.CreateLoad(RedudantAllocation[Index]), MetaOutput);
    Builder.SetInsertPoint(SecondElseTerm);
  }

/*

  //Instruction* OneCmp = dyn_cast<Instruction>(Builder.CreateICmpEQ(RedundantID, PassAuxiliary->One32Bit));
  //SplitBlockAndInsertIfThenElse(OneCmp, OneCmp->getNextNode(), &ElseIfTerm, &SecondElseTerm); 
  //Builder.SetInsertPoint(ElseIfTerm);
  //Builder.CreateStore(Builder.CreateLoad(FirstRedundant), MetaOutput);

  Builder.SetInsertPoint(SecondElseTerm);
  RedundantID = Builder.CreateLoad(RedundantIDAddr);
  Instruction* TwoCmp = dyn_cast<Instruction>(Builder.CreateICmpEQ(RedundantID, PassAuxiliary->Two32Bit));
  Instruction* NewBranch  = SplitBlockAndInsertIfThen(TwoCmp, TwoCmp->getNextNode(), false);
  Builder.SetInsertPoint(NewBranch);
  Builder.CreateStore(Builder.CreateLoad(SecondRedundant), MetaOutput);
*/



}  



       
inline void CudaConfigure(CallInst *FunctionCall, Value* CudaConfigurations[8]) {
  for(int ArgIndex = 0; ArgIndex < 4; ArgIndex++){
    LoadInst* LoadInstruction = dyn_cast_or_null<LoadInst>(FunctionCall->getArgOperand(ArgIndex));
    GetElementPtrInst* GEP = dyn_cast_or_null<GetElementPtrInst>(LoadInstruction->getPointerOperand());
    AllocaInst* AllocaCoerce  = dyn_cast_or_null<AllocaInst>(GEP->getPointerOperand());
    for(Use& U : AllocaCoerce->uses()){
      User* User = U.getUser();
      BitCastInst* Casted = dyn_cast_or_null<BitCastInst>(User);
      if(Casted != nullptr){
        Instruction* TempInstruction = Casted;
        while(dyn_cast_or_null<MemCpyInst>(TempInstruction) == nullptr)
          TempInstruction = TempInstruction->getNextNode();
        MemCpyInst* DimensionMemoryCopy = dyn_cast_or_null<MemCpyInst>(TempInstruction);
        BitCastInst* CopySrc = dyn_cast_or_null<BitCastInst>(DimensionMemoryCopy->getArgOperand(1));
        AllocaInst* Alloca  = dyn_cast_or_null<AllocaInst>(CopySrc->getOperand(0));
        TempInstruction = Alloca;
        Instruction* PrevInst = Alloca->getPrevNode();
        while(true){  
          if(MemCpyInst* MemoryInstruction = dyn_cast_or_null<MemCpyInst>(TempInstruction)){
            BitCastInst* PossibleBitCast = dyn_cast_or_null<BitCastInst>(MemoryInstruction->getArgOperand(0));
            AllocaInst* PossibleAllocation  = dyn_cast_or_null<AllocaInst>(PossibleBitCast->getOperand(0));
            if(PossibleAllocation == Alloca){
              BitCastInst* CopyDestination = dyn_cast_or_null<BitCastInst>(MemoryInstruction->getArgOperand(1));
              AllocaInst* DestinationAllocation  = dyn_cast_or_null<AllocaInst>(CopyDestination->getOperand(0));
              TempInstruction = DestinationAllocation;
              while(true){
                if(CallInst* DimensionCall = dyn_cast_or_null<CallInst>(TempInstruction)){
                  StringRef FunctionName = DimensionCall->getCalledFunction()->getName();
                  if(FunctionName.contains("dim3") && 
                      DimensionCall->getArgOperand(0) == DestinationAllocation){
                        CudaConfigurations[ArgIndex] = DimensionCall->getArgOperand(ArgIndex%2 + 1);
                        CudaConfigurations[ArgIndex+4] = DimensionCall;
                        errs() << ArgIndex << *DimensionCall << "\n";
                    break;
                  }
                }
                PrevInst = TempInstruction;
                TempInstruction = TempInstruction->getNextNode();
                if(TempInstruction == nullptr) TempInstruction = PrevInst->getParent()->getNextNode()->getFirstNonPHI();

              }
              break;
            }
          }
          PrevInst = TempInstruction;
          TempInstruction = TempInstruction->getNextNode();
          if(TempInstruction == nullptr) TempInstruction = PrevInst->getParent()->getNextNode()->getFirstNonPHI();
        }
      }
    }
  }
}

inline FunctionType* getTheNewKernelFunctionType(std::vector<Value *> Args, Type* ReturnType){

  std::vector<Type * > NewKernelTypes;
  for(size_t ArgIndex = 0; ArgIndex < Args.size(); ArgIndex++){
    NewKernelTypes.push_back(Args.at(ArgIndex)->getType());
  }

  return FunctionType::get(ReturnType, NewKernelTypes, true);
}

inline bool isReplicate(CallInst *FunctionCall) {
  return FunctionCall->hasMetadata(RedundantString);
}

inline StringRef getMetadataString(CallInst* FunctionCall) {
  return cast<MDString>(FunctionCall->getMetadata(RedundantString)->getOperand(0))->getString();
}


 inline std::vector<std::string> parseData(StringRef MetaData) {
    std::vector<std::string> Returns;
    std::string MetaDataString = MetaData.str();
    std::string Data = "";
    bool Delimiter = true;
    for(size_t Index = 0; Index < MetaDataString.size(); Index++){
        Data += MetaDataString[Index];

        if(MetaDataString[Index] == ' ' || Index == MetaDataString.size() - 1){
            if(!Delimiter)
                Returns.push_back(Data.substr(1,std::string::npos));
            
            Delimiter = !Delimiter;
            Data = "";
        }
    }

    return Returns;

  }

inline bool isForLoop(BasicBlock* PrevBB){
    bool IsLoop = false;
    while(PrevBB != nullptr){
      StringRef BBName = PrevBB->getName();
      IsLoop = BBName.contains("for");
      if(IsLoop){
        break;
      }
      PrevBB = PrevBB->getPrevNode();
      
    }
    return IsLoop;
}

inline void parseOutput(std::vector<CallInst *> CudaMallocFunctionCalls, Output* SingleOutput){
  std::string VariableName = SingleOutput->Name;
  AllocaInst *AllocaVariable = nullptr;
  Type *DestinationType = nullptr;
  for (size_t Index = 0; Index < CudaMallocFunctionCalls.size(); Index++) {
      CallInst *CudaMallocFunctionCall = CudaMallocFunctionCalls.at(Index);
      Value* Operand = CudaMallocFunctionCall->getArgOperand(0);
      if (BitCastInst *BitCastVariable = dyn_cast<BitCastInst>(Operand)) {
        AllocaVariable = dyn_cast<AllocaInst>(BitCastVariable->getOperand(0));
        DestinationType = dyn_cast<PointerType>(BitCastVariable->getDestTy());
      }else if(dyn_cast_or_null<AllocaInst>(Operand) != nullptr){
         AllocaVariable = dyn_cast<AllocaInst>(Operand);
      }
      std::string OutputName = AllocaVariable->getName().str();
      if (VariableName == OutputName) {
        SingleOutput->OutputAllocation = AllocaVariable;
        SingleOutput->OutputType = AllocaVariable->getAllocatedType();
        SingleOutput->DestinationType = DestinationType;
        SingleOutput->MallocInstruction = CudaMallocFunctionCall;
        break;
      }
  }

}



inline CallInst* getTheMemoryFunction(std::vector<CallInst *> Functions, StringRef OutputName, Output* SingleOutput = nullptr){
  CallInst* ReturnFuntion = nullptr;
  for(size_t Index = 0; Index < Functions.size(); Index++){
    CallInst* CurrentFuntion = Functions.at(Index);
    BitCastInst* Bitcasted = dyn_cast<BitCastInst>(CurrentFuntion->getOperand(0));
    LoadInst* Loaded = dyn_cast_or_null<LoadInst>(Bitcasted->getOperand(0));
    AllocaInst* Alloca = nullptr;
    if(Loaded == nullptr){
      Alloca = dyn_cast<AllocaInst>(Bitcasted->getOperand(0));
      }
    else
      Alloca = dyn_cast<AllocaInst>(Loaded->getOperand(0));
      
    if(OutputName.contains(Alloca->getName())){
      ReturnFuntion = CurrentFuntion;
      if(SingleOutput != nullptr){
        SingleOutput->DestinationType = Bitcasted->getDestTy();
        SingleOutput->OutputAllocation = Alloca;
        SingleOutput->OutputType = Alloca->getAllocatedType();
        SingleOutput->MallocInstruction = ReturnFuntion;
        SingleOutput->Name = Alloca->getName();
        SingleOutput->Size = ReturnFuntion->getArgOperand(1);
        ConstantInt* CI = dyn_cast<ConstantInt>(SingleOutput->Size);
        PointerType* OutputType = dyn_cast<PointerType>(SingleOutput->OutputType);
        
        //getScalarSizeInBits() ;
        SingleOutput->NumberOfElement = ConstantInt::get( Type::getInt64Ty(ReturnFuntion->getContext()), CI->getSExtValue() / (OutputType->getElementType()->getScalarSizeInBits()/8));
        

      }
      break;
    }
  }
  return ReturnFuntion;
}
inline BasicBlock* createCheckPoint(IRBuilder<> Builder, Output* OutputToReplicate, BasicBlock* InsertBefore, Auxiliary* PassAuxiliary){
  LLVMContext& Context = InsertBefore->getContext();
  BasicBlock* NewBasicBlock = BasicBlock::Create(Context, "Checkpoint", InsertBefore->getParent(), InsertBefore);
  Builder.SetInsertPoint(NewBasicBlock);
  Value* Size = OutputToReplicate->Size;
  Type* OutputType = OutputToReplicate->OutputType;

  Instruction* CheckpointAllocation = Builder.CreateAlloca(OutputToReplicate->OutputType, 0, "Output_Checkpoint");
  //errs() << *PassAuxiliary->MallocFunction.getCallee() << "\n";

  Instruction* MallocCall = Builder.CreateCall(PassAuxiliary->MallocFunction, {Size});
  Value* Bitcasted = Builder.CreateBitCast(MallocCall, OutputType);
  Builder.CreateStore(Bitcasted, CheckpointAllocation);

  Instruction* LoadedCheckpoint = Builder.CreateLoad(CheckpointAllocation);
  Value* BitcastedCheckpoint = Builder.CreateBitCast(LoadedCheckpoint, PassAuxiliary->Int8PtrType);

  Instruction* LoadedOutput = Builder.CreateLoad(OutputToReplicate->OutputAllocation);
  Value* BitcastedOutput = Builder.CreateBitCast(LoadedOutput, PassAuxiliary->Int8PtrType);
  
 Builder.CreateCall(PassAuxiliary->CudaMemCopy, {BitcastedCheckpoint, BitcastedOutput, Size, PassAuxiliary->Two32Bit});

  Builder.CreateBr(InsertBefore);

  OutputToReplicate->Checkpoint = CheckpointAllocation;

  return NewBasicBlock;
}

inline BasicBlock* createErrorValue(IRBuilder<> Builder, BasicBlock* InsertBefore, Auxiliary* PassAuxiliary, std::pair<Instruction*, Instruction* > &Errors){
  LLVMContext& Context = InsertBefore->getContext();
  BasicBlock* NewBasicBlock = BasicBlock::Create(Context,"Errors", InsertBefore->getParent(), InsertBefore);
  Builder.SetInsertPoint(NewBasicBlock);
  Instruction* ErrorAllocation = Builder.CreateAlloca(PassAuxiliary->Int32PtrType, 0, "ErrorCPU");
  Instruction* MallocCall = Builder.CreateCall(PassAuxiliary->MallocFunction, {PassAuxiliary->Four64Bit});
  Value* Bitcasted = Builder.CreateBitCast(MallocCall, PassAuxiliary->Int32PtrType);
  Builder.CreateStore(Bitcasted, ErrorAllocation);

  Instruction* ErrorGPUAllocation = Builder.CreateAlloca(PassAuxiliary->Int32PtrType, 0, "ErrorGPU");
  Value* BitcastedGPU = Builder.CreateBitCast(ErrorGPUAllocation, PointerType::get(PassAuxiliary->Int8PtrType, 0));
  Builder.CreateCall(PassAuxiliary->CudaMalloc, {BitcastedGPU, PassAuxiliary->Four64Bit});

  Instruction* LoadedErrorCPU = Builder.CreateLoad(ErrorAllocation);
  Value* BitcastedErrorCPU= Builder.CreateBitCast(LoadedErrorCPU, PassAuxiliary->Int8PtrType);
  Instruction* LoadedErrorGPU = Builder.CreateLoad(ErrorGPUAllocation);
  Value* BitcastedErrorGPU= Builder.CreateBitCast(LoadedErrorGPU, PassAuxiliary->Int8PtrType);

  Builder.CreateCall(PassAuxiliary->CudaMemCopy, {BitcastedErrorGPU, BitcastedErrorCPU, PassAuxiliary->Four64Bit, PassAuxiliary->One32Bit});
  Builder.CreateBr(InsertBefore);

  Errors = std::make_pair(ErrorAllocation, ErrorGPUAllocation);
  return NewBasicBlock;
}

inline BasicBlock* createAndAllocateVariableAndreMemCpy(IRBuilder<> Builder, Output* OutputToReplicate, BasicBlock* InsertBefore, Auxiliary* PassAuxiliary, int RedundantIndex){
  LLVMContext &Context = InsertBefore->getContext();
  Type* OutputType = OutputToReplicate->OutputType;
  Type *Int32Type = PassAuxiliary->Int32Type;
  Type *Int8PtrType = PassAuxiliary->Int8PtrType;
  Type* DestinationType = OutputToReplicate->DestinationType;

  Value* Three32Bit =  PassAuxiliary->Three32Bit;
  Value* OriginalOutput = OutputToReplicate->OutputAllocation;
  Value* Size = OutputToReplicate->MallocInstruction->getArgOperand(1);

  BasicBlock* NewBasicBlock = BasicBlock::Create(Context,"RedundantOutput", InsertBefore->getParent(), InsertBefore);
  Builder.SetInsertPoint(NewBasicBlock);
  
  Instruction* NewAllocated = Builder.CreateAlloca(OutputType, nullptr, OutputToReplicate->Name);
  
  Instruction* ClonedMalloc = OutputToReplicate->MallocInstruction->clone();
  CallInst* Cloned = dyn_cast<CallInst>(ClonedMalloc);
  if(DestinationType == nullptr){
    Cloned->setArgOperand(0, NewAllocated);
  }else{
    Value* BitcastedCloned = Builder.CreateBitCast(NewAllocated, DestinationType);
    Cloned->setArgOperand(0, BitcastedCloned);
  }

  Instruction* InsertLocation = dyn_cast<Instruction>(Cloned->getOperand(0));
  Cloned->insertAfter(InsertLocation);
  Value* BitcastedCloned = Builder.CreateBitCast(Builder.CreateLoad(NewAllocated), Int8PtrType);
  Value* LoadedOutput = Builder.CreateLoad(OriginalOutput);
  Value* BitcastedOutput = Builder.CreateBitCast(LoadedOutput, Int8PtrType);
  Builder.CreateCall(PassAuxiliary->CudaMemCopy, {BitcastedCloned, BitcastedOutput, Size, Three32Bit});
  
  Builder.CreateBr(InsertBefore);
  OutputToReplicate->Replications[RedundantIndex] = NewAllocated;
  
  return NewBasicBlock;
}

inline void createAndAllocateVariableAndreMemCpy(IRBuilder<> Builder, Output* OutputToReplicate, Instruction& LastInstructionOfPrevBB, FunctionCallee CudaMemcpy, bool IsLoop, int RedundantNumber){
  LLVMContext &Context = LastInstructionOfPrevBB.getContext();
  Type *Int32Type = Type::getInt32Ty(Context);
  Type *Int8PtrType = Type::getInt8PtrTy(Context);
  Value* Three32Bit =  ConstantInt::get(Int32Type, 3);
  AllocaInst* OutputToReplicateAllocation = OutputToReplicate->OutputAllocation;
  Value* Size = OutputToReplicate->MallocInstruction->getArgOperand(1);
  Type* OutputType = OutputToReplicate->OutputType;
  Type* DestinationType = OutputToReplicate->DestinationType;
  for(int Replication = 0; Replication < RedundantNumber; Replication++){
    Builder.SetInsertPoint(OutputToReplicateAllocation->getNextNode());       
    Instruction* NewAllocated = Builder.CreateAlloca(OutputType, nullptr,  OutputToReplicate->Name);
    Instruction* ClonedMalloc = OutputToReplicate->MallocInstruction->clone();
    CallInst* Cloned = dyn_cast<CallInst>(ClonedMalloc);
    if(DestinationType == nullptr){
      Cloned->setArgOperand(0, NewAllocated);
    }else{
      Value* BitcastedCloned = Builder.CreateBitCast(NewAllocated, DestinationType);
      Cloned->setArgOperand(0, BitcastedCloned);
    }
    Cloned->insertAfter(OutputToReplicate->MallocInstruction);
    Builder.SetInsertPoint(&LastInstructionOfPrevBB); 
    Value* BitcastedCloned = Builder.CreateBitCast(Builder.CreateLoad(NewAllocated), Int8PtrType);
    Value* LoadedOutput = Builder.CreateLoad(OutputToReplicateAllocation);
    Value* BitcastedOutput = Builder.CreateBitCast(LoadedOutput, Int8PtrType);
    Builder.CreateCall(CudaMemcpy, {BitcastedCloned, BitcastedOutput, Size, Three32Bit});
    OutputToReplicate->Replications[Replication] = NewAllocated;
  }
}


inline Function *createMajorityVoting(Module& M, PointerType *MajorityVotingPointerType, Auxiliary* PassAuxiliary, std::string MajorityVotingFunctionName) {

  // std::to_string(MajorityVotingPointerType->getTypeID());

  

  Function *CudaGlobalRegister = PassAuxiliary->CudaGlobalRegisterFunction;
  Function *CudaRegister = PassAuxiliary->CudaRegisterFunction;
  Function *CudaSetupArgument = PassAuxiliary->CudaSetupArgument;
  Function *CudaLaunch = PassAuxiliary->CudaLaunch;

  Type *Int64Type = PassAuxiliary->Int64Type;
  Value *Zero32bit = PassAuxiliary->Zero32Bit;

  PointerType *Int8PtrType = PassAuxiliary->Int8PtrType;

  FunctionCallee MajorityVotingCallee = M.getOrInsertFunction(MajorityVotingFunctionName, PassAuxiliary->VoidType, MajorityVotingPointerType, MajorityVotingPointerType,MajorityVotingPointerType, Int64Type);

  std::vector<Value *> Parameters;
  Function *MajorityVotingFunction = dyn_cast<Function>(MajorityVotingCallee.getCallee());

  MajorityVotingFunction->setCallingConv(CallingConv::C);
  Function::arg_iterator Args = MajorityVotingFunction->arg_begin();

  Value *A = Args++; A->setName("A");

  Value *B = Args++; B->setName("B");

  Value *C = Args++; C->setName("C");

  Value *Size = Args++; Size->setName("Size");

  BasicBlock *EntryBlock = BasicBlock::Create(M.getContext(), "entry", MajorityVotingFunction);

  IRBuilder<> Builder(EntryBlock); Builder.SetInsertPoint(EntryBlock);

  Value *Aptr = Builder.CreateAlloca(MajorityVotingPointerType, nullptr, "A.addr");

  Value *Bptr = Builder.CreateAlloca(MajorityVotingPointerType, nullptr, "B.addr");

  Value *Cptr = Builder.CreateAlloca(MajorityVotingPointerType, nullptr, "C.addr");

  Value *Sizeptr = Builder.CreateAlloca(Int64Type, nullptr, "size.addr");

  Builder.CreateStore(A, Aptr);
  Builder.CreateStore(B, Bptr);
  Builder.CreateStore(C, Cptr);
  Builder.CreateStore(Size, Sizeptr);

  Parameters.push_back(Aptr);
  Parameters.push_back(Bptr);
  Parameters.push_back(Cptr);
  Parameters.push_back(Sizeptr);

  int Offset = 0;
  int SizeParameter = 8;
  for (unsigned long Index = 0; Index < Parameters.size(); Index++) {
    Value *Parameter = Parameters.at(Index);
    Value *BitcastParameter = Builder.CreateBitCast(Parameter, Int8PtrType);
    Value *OffsetValue = ConstantInt::get(Int64Type, Offset);

    if(dyn_cast_or_null<PointerType>(Parameter->getType()) == nullptr) SizeParameter = 4;
    else SizeParameter = 8;

    Value *SizeValue = ConstantInt::get(Int64Type, SizeParameter);
    Value *CudaSetupArgumentCall = Builder.CreateCall( CudaSetupArgument, {BitcastParameter, SizeValue, OffsetValue});
    Instruction *IsError = dyn_cast<Instruction>( Builder.CreateICmpEQ(CudaSetupArgumentCall, Zero32bit));
    if (Index == 0) Builder.CreateRetVoid(); 

    Instruction *SplitPoint = SplitBlockAndInsertIfThen(IsError, IsError->getNextNode(), false);

    SplitPoint->getParent()->setName("setup.next");

    Builder.SetInsertPoint(SplitPoint);
    Offset += SizeParameter;
  }

  Builder.CreateCall(CudaLaunch, {Builder.CreateBitCast( MajorityVotingFunction , Int8PtrType)});
  registerTheFunction(MajorityVotingFunction, PassAuxiliary);

  return MajorityVotingFunction;
}

inline void createOrInsertMajorityVotingFunction(Module& M, Output* OutputObject, Auxiliary* PassAuxiliary){
  Type* OutputType = OutputObject->OutputType;
  std::string MajorityFunctionName = "majorityVoting" + std::to_string(OutputType->getPointerElementType()->getTypeID()); // 
  //MajorityFunctionName = "_Z14majorityVotingPfS_S_l" ; 
  Function* MajorityFunction = M.getFunction(MajorityFunctionName);

  if(MajorityFunction == nullptr) MajorityFunction = createMajorityVoting(M, dyn_cast<PointerType>(OutputType), PassAuxiliary, MajorityFunctionName);
  OutputObject->MajorityVotingFunction = MajorityFunction;
}


inline std::vector<Function * > getValidKernels(NamedMDNode *Annotations){
  std::vector<Function *> ValidKernels;


  for(size_t Index = 0; Index < Annotations->getNumOperands(); Index++){
    MDNode* SingleAnnotation = Annotations->getOperand(Index);
    MDString* Operand = dyn_cast<MDString>(SingleAnnotation->getOperand(1));
    if(Operand->getString() == "kernel"){
      ValueAsMetadata* VM = dyn_cast<ValueAsMetadata>(SingleAnnotation->getOperand(0));
      Function* ValidFunction = dyn_cast<Function>(VM->getValue());
      if(!ValidFunction->getName().contains(RevisitedSuffix))
        ValidKernels.push_back(ValidFunction);
    }
  }

  return ValidKernels;

}

inline Function* createDeviceMajorityVotingFunction(Module& M, Auxiliary* PassAuxiliary, PointerType* OutputPointerType, std::string MajorityFunctionName){
  LLVMContext& Context = M.getContext();
  FunctionCallee MajorityVotingCallee = M.getOrInsertFunction(MajorityFunctionName, PassAuxiliary->VoidType, OutputPointerType, OutputPointerType, OutputPointerType, PassAuxiliary->Int64Type);
  
  Function* MajorityVotingFunction = dyn_cast<Function>(MajorityVotingCallee.getCallee());
  MajorityVotingFunction->setCallingConv(CallingConv::C);

  BasicBlock* EntryBlock = BasicBlock::Create(Context, "entry", MajorityVotingFunction);
  BasicBlock *IfBlock = BasicBlock::Create(Context, "If", MajorityVotingFunction);
  BasicBlock *SecondIfBlock = BasicBlock::Create(Context, "If", MajorityVotingFunction);
  BasicBlock *TailBlock = BasicBlock::Create(Context, "Tail", MajorityVotingFunction);


  IRBuilder<> Builder(EntryBlock);    
  Function::arg_iterator Args = MajorityVotingFunction->arg_begin();
  Value *DeviceA = Args++;
  DeviceA->setName("data1");
  Value *DeviceB = Args++;
  DeviceB->setName("data2");
  Value *DeviceC = Args++;
  DeviceC->setName("data3");
  Value *ArraySize = Args;
  ArraySize->setName("size");

  AllocaInst *DeviceAAllocation = Builder.CreateAlloca(OutputPointerType, nullptr, "data1.addr");
  AllocaInst *DeviceBAllocation = Builder.CreateAlloca(OutputPointerType, nullptr, "data2.addr");
  AllocaInst *DeviceCAllocation = Builder.CreateAlloca(OutputPointerType, nullptr, "data3.addr");
  AllocaInst *DeviceArraySizeAllocation = Builder.CreateAlloca(PassAuxiliary->Int64Type, nullptr, "size.addr");
  //Value *ThreadIDptr = Builder.CreateAlloca(PassAuxiliary->Int32Type, nullptr, "i");

  Value *ThreadIDXptr = Builder.CreateAlloca(PassAuxiliary->Int32Type, nullptr, "ThreadIDX");
  Value *ThreadIDYptr = Builder.CreateAlloca(PassAuxiliary->Int32Type, nullptr, "ThreadIDY");
  Value *ThreadIDptr = Builder.CreateAlloca(PassAuxiliary->Int32Type, nullptr, "ThreadID");

  Builder.CreateStore(DeviceA, DeviceAAllocation);
  Builder.CreateStore(DeviceB, DeviceBAllocation);
  Builder.CreateStore(DeviceC, DeviceCAllocation);
  Builder.CreateStore(ArraySize, DeviceArraySizeAllocation);

  Value *BlockID = Builder.CreateCall(PassAuxiliary->CudaDimensionFunctions[0]);
  Value *BlockDimension = Builder.CreateCall(PassAuxiliary->CudaDimensionFunctions[4]);
  Value *BlockXGrid = Builder.CreateMul(BlockDimension, BlockID,"mul");
  Value *ThreadXNumber = Builder.CreateCall(PassAuxiliary->CudaDimensionFunctions[2]);
  Value *ThreadIDX = Builder.CreateAdd(BlockXGrid, ThreadXNumber, "add");
  Builder.CreateStore(ThreadIDX, ThreadIDXptr);

  Value *BlockIDY = Builder.CreateCall(PassAuxiliary->CudaDimensionFunctions[1]);
  Value *BlockDimensionY = Builder.CreateCall(PassAuxiliary->CudaDimensionFunctions[5]);
  Value *BlockXGridY = Builder.CreateMul(BlockIDY, BlockDimensionY,"mul");
  Value *ThreadNumberY = Builder.CreateCall(PassAuxiliary->CudaDimensionFunctions[3]);
  Value *ThreadIDY = Builder.CreateAdd(BlockXGridY, ThreadNumberY, "add");
  Builder.CreateStore(ThreadIDY, ThreadIDYptr);

  Value *SecondBlockDimension = Builder.CreateCall(PassAuxiliary->CudaDimensionFunctions[4]);
  Value *ThreadIDYptrXSecondBlockDimension = Builder.CreateMul(SecondBlockDimension, Builder.CreateLoad(ThreadIDYptr),"mul");
  Value *ThreadID = Builder.CreateAdd(ThreadIDYptrXSecondBlockDimension, Builder.CreateLoad(ThreadIDXptr), "add");
  Builder.CreateStore(ThreadID, ThreadIDptr);

  Value *TID = Builder.CreateLoad(ThreadIDptr);
  Value *Extented = Builder.CreateSExt(TID, PassAuxiliary->Int64Type);
  Value *ArraySizeValue = Builder.CreateLoad(DeviceArraySizeAllocation);

  Value *SizeTIDCMP = Builder.CreateICmpSLT(Extented, ArraySizeValue);
  Builder.CreateCondBr(SizeTIDCMP, IfBlock, TailBlock);

  Builder.SetInsertPoint(IfBlock);
  Value *DeviceBPointer = Builder.CreateLoad(DeviceBAllocation);

  TID = Builder.CreateLoad(ThreadIDptr);
  Value *TID64Bit = Builder.CreateSExt(TID, PassAuxiliary->Int64Type);
  Value *PointerToTIDthElementOfDeviceB = Builder.CreateInBoundsGEP(DeviceBPointer, TID64Bit);

  Value *TIDthElementOfDeviceB = Builder.CreateLoad(PointerToTIDthElementOfDeviceB);

  Value *DeviceCPointer = Builder.CreateLoad(DeviceCAllocation);
  TID = Builder.CreateLoad(ThreadIDptr);
  TID64Bit = Builder.CreateSExt(TID, PassAuxiliary->Int64Type);
  Value *PointerToTIDthElementOfDeviceC = Builder.CreateInBoundsGEP(DeviceCPointer, TID64Bit);
  Value *TIDthElementOfDeviceC = Builder.CreateLoad(PointerToTIDthElementOfDeviceC);
  Value *DeviceADeviceBCMP;

  if( TIDthElementOfDeviceC->getType()->isFloatTy() || TIDthElementOfDeviceB->getType()->isDoubleTy())
   DeviceADeviceBCMP = Builder.CreateFCmpUNE(TIDthElementOfDeviceC, TIDthElementOfDeviceB);
  else
    DeviceADeviceBCMP = Builder.CreateICmpNE(TIDthElementOfDeviceC, TIDthElementOfDeviceB);

  Builder.CreateCondBr(DeviceADeviceBCMP, SecondIfBlock, TailBlock);

  Builder.SetInsertPoint(SecondIfBlock);

  DeviceCPointer = Builder.CreateLoad(DeviceCAllocation);
  TID = Builder.CreateLoad(ThreadIDptr);
  TID64Bit = Builder.CreateSExt(TID, PassAuxiliary->Int64Type);
  PointerToTIDthElementOfDeviceC = Builder.CreateInBoundsGEP(DeviceCPointer, TID64Bit);
  TIDthElementOfDeviceC = Builder.CreateLoad(PointerToTIDthElementOfDeviceC);



  Value *DeviceAPointer = Builder.CreateLoad(DeviceAAllocation);

  TID = Builder.CreateLoad(ThreadIDptr);
  TID64Bit = Builder.CreateSExt(TID, PassAuxiliary->Int64Type);
  Value *PointerToTIDthElementOfDeviceA = Builder.CreateInBoundsGEP(DeviceAPointer, TID64Bit);
  Builder.CreateStore(TIDthElementOfDeviceC, PointerToTIDthElementOfDeviceA);

  Builder.CreateBr(TailBlock);

  Builder.SetInsertPoint(TailBlock);
  Builder.CreateRetVoid();

  return MajorityVotingFunction;
}


inline void createHostRevisited(Function* NewKernelFunction, Function* OriginalFunction, Auxiliary* PassAuxiliary, int RedundanyNumber){
  Value* RedundantArgs[RedundanyNumber];
  Instruction* RedundantAllocations[RedundanyNumber];

  ValueToValueMapTy VMap;
  SmallVector<ReturnInst*, 8> Returns;
  Function::arg_iterator DestI = NewKernelFunction->arg_begin();
  for (const Argument & I : OriginalFunction->args())
    if (VMap.count(&I) == 0) {    
      DestI->setName(I.getName()); 
      VMap[&I] = &*DestI++;       
    }
    
  CloneFunctionInto(NewKernelFunction,OriginalFunction,VMap,false,Returns);

  Function::arg_iterator Args = NewKernelFunction->arg_end();
  Args--;
  for(int Index = 0; Index < RedundanyNumber; Index++){
    RedundantArgs[Index] = Args--;
  }
  Value *OriginalBased = Args--;
  Type *OutputType = RedundantArgs[0]->getType();
  Type *BasedType = OriginalBased->getType();

  BasicBlock* FirstBB = dyn_cast<BasicBlock>(NewKernelFunction->begin());
  BasicBlock* LastBB = &(NewKernelFunction->back());
  BasicBlock* LauchCallBB = LastBB->getPrevNode();
  BasicBlock* LastSetupBB = LauchCallBB->getPrevNode();
  Instruction *Inst = dyn_cast<Instruction>(FirstBB->begin());
  IRBuilder<> Builder(Inst);

  while(dyn_cast_or_null<AllocaInst>(Inst) != nullptr)
    Inst = Inst->getNextNode();

  Builder.SetInsertPoint(Inst);
  for(int Index = 0; Index < RedundanyNumber; Index++){
    RedundantAllocations[Index] = Builder.CreateAlloca(OutputType, nullptr, "RedundantArg");
  }
  //Instruction* FirstRedundantAllocation = Builder.CreateAlloca(OutputType,nullptr, "FirstRedundantArg");
  //Instruction* SecondRedundantAllocation = Builder.CreateAlloca(OutputType,nullptr, "SecondRedundantArg");
  Instruction* BasedAllocation = Builder.CreateAlloca(BasedType,nullptr, "OriginalBased");

  while(dyn_cast_or_null<StoreInst>(Inst) != nullptr)
    Inst = Inst->getNextNode();

  Builder.SetInsertPoint(Inst);
  for(int Index = 0; Index < RedundanyNumber; Index++){
    Builder.CreateStore(RedundantArgs[Index],RedundantAllocations[Index]);
  }
  Builder.CreateStore(OriginalBased,BasedAllocation);

  Inst = &(LauchCallBB->back());
  while(dyn_cast_or_null<CallInst>(Inst) == nullptr)
    Inst = Inst->getPrevNode();
  CallInst* CudaLaunchCall = dyn_cast<CallInst>(Inst);
  
  CudaLaunchCall->setArgOperand(0, Builder.CreateBitCast(NewKernelFunction, PassAuxiliary->Int8PtrType));
  
  Inst = &(LastSetupBB->back());
  while(dyn_cast_or_null<CallInst>(Inst) == nullptr)
    Inst = Inst->getPrevNode();

  ConstantInt* Offset = dyn_cast<ConstantInt>(Inst->getOperand(2));
  ConstantInt* ArgumentSize =  dyn_cast<ConstantInt>((OutputType->isPointerTy()) ? PassAuxiliary->Eight64Bit : PassAuxiliary->Four64Bit);
  int64_t OffsetValue = Offset->getSExtValue() + ArgumentSize->getSExtValue();
  Builder.SetInsertPoint(CudaLaunchCall);
  
  Value* CastedArgument = Builder.CreateBitCast(BasedAllocation, PassAuxiliary->Int8PtrType);
  ArgumentSize =  dyn_cast<ConstantInt>((BasedType->isPointerTy()) ? PassAuxiliary->Eight64Bit : PassAuxiliary->Four64Bit);
  Value* SetupCall = Builder.CreateCall(PassAuxiliary->CudaSetupArgument,{CastedArgument, ArgumentSize, ConstantInt::get(PassAuxiliary->Int64Type, OffsetValue)});
  Instruction* Condition = dyn_cast<Instruction>(Builder.CreateICmpEQ(SetupCall, PassAuxiliary->Zero32Bit));
  BranchInst* Branch = dyn_cast<BranchInst>(Condition->getNextNode());
  Instruction* NewBBInstruction = SplitBlockAndInsertIfThen(Condition, Condition->getNextNode(), false);
  Branch = dyn_cast<BranchInst>(Condition->getNextNode());
  Branch->setOperand(1, LastBB);
  
  Builder.SetInsertPoint(NewBBInstruction);
  ArgumentSize =  dyn_cast<ConstantInt>((OutputType->isPointerTy()) ? PassAuxiliary->Eight64Bit : PassAuxiliary->Four64Bit);
  for(int Index = 0; Index < RedundanyNumber; Index++){
    CastedArgument = Builder.CreateBitCast(RedundantAllocations[Index], PassAuxiliary->Int8PtrType);
    OffsetValue += 8;
    SetupCall = Builder.CreateCall(PassAuxiliary->CudaSetupArgument,{CastedArgument, ArgumentSize, ConstantInt::get(PassAuxiliary->Int64Type, OffsetValue)});
    Condition = dyn_cast<Instruction>(Builder.CreateICmpEQ(SetupCall, PassAuxiliary->Zero32Bit));
    NewBBInstruction = SplitBlockAndInsertIfThen(Condition,  Condition->getNextNode(), false);
    Branch = dyn_cast<BranchInst>(Condition->getNextNode());
    Branch->setOperand(1, LastBB);
    Builder.SetInsertPoint(NewBBInstruction);

  }

  /*
  //CastedArgument = Builder.CreateBitCast(FirstRedundantAllocation, PassAuxiliary->Int8PtrType);
  //OffsetValue += 8;
  //SetupCall = Builder.CreateCall(PassAuxiliary->CudaSetupArgument,{CastedArgument, ArgumentSize, ConstantInt::get(PassAuxiliary->Int64Type, OffsetValue)});
  //Condition = dyn_cast<Instruction>(Builder.CreateICmpEQ(SetupCall, PassAuxiliary->Zero32Bit));
  //NewBBInstruction = SplitBlockAndInsertIfThen(Condition,  Condition->getNextNode(), false);

  Branch = dyn_cast<BranchInst>(Condition->getNextNode());
  Branch->setOperand(1, LastBB);



  CastedArgument = Builder.CreateBitCast(SecondRedundantAllocation, PassAuxiliary->Int8PtrType);
  OffsetValue += ArgumentSize->getSExtValue();
  SetupCall = Builder.CreateCall(PassAuxiliary->CudaSetupArgument,{CastedArgument, ArgumentSize, ConstantInt::get(PassAuxiliary->Int64Type, OffsetValue)});
  Condition = dyn_cast<Instruction>(Builder.CreateICmpEQ(SetupCall, PassAuxiliary->Zero32Bit));
  Branch = dyn_cast<BranchInst>(Condition->getNextNode());
  NewBBInstruction = SplitBlockAndInsertIfThen(Condition, Condition->getNextNode(), false);
  Branch = dyn_cast<BranchInst>(Condition->getNextNode());
  Branch->setOperand(1, LastBB);
*/

  registerTheFunction(NewKernelFunction, PassAuxiliary);
  
}


inline Function* createDeviceDetectionFunction(Module& M, Auxiliary* PassAuxiliary, PointerType* OutputPointerType, std::string DetectionFunctionName){
  LLVMContext& Context = M.getContext();
  FunctionCallee DetectionCallee = M.getOrInsertFunction(DetectionFunctionName, PassAuxiliary->VoidType, OutputPointerType, OutputPointerType, PassAuxiliary->Int64Type, PassAuxiliary->Int32PtrType);


  Function* DetectionFunction = dyn_cast<Function>(DetectionCallee.getCallee());
  DetectionFunction->setCallingConv(CallingConv::C);

  BasicBlock* EntryBlock = BasicBlock::Create(Context, "entry", DetectionFunction);
  BasicBlock *IfBlock = BasicBlock::Create(Context, "If", DetectionFunction);
  BasicBlock *SecondIfBlock = BasicBlock::Create(Context, "If", DetectionFunction);
  BasicBlock *TailBlock = BasicBlock::Create(Context, "Tail", DetectionFunction);


  IRBuilder<> Builder(EntryBlock);    
  Function::arg_iterator Args = DetectionFunction->arg_begin();
  Value *DeviceA = Args++;
  DeviceA->setName("data1");
  Value *DeviceB = Args++;
  DeviceB->setName("data2");
  Value *ArraySize = Args++;
  ArraySize->setName("size");
  Value *Error = Args;
  Error->setName("Error");

  AllocaInst *DeviceAAllocation = Builder.CreateAlloca(OutputPointerType, nullptr, "data1.addr");
  AllocaInst *DeviceBAllocation = Builder.CreateAlloca(OutputPointerType, nullptr, "data2.addr");
  AllocaInst *DeviceArraySizeAllocation = Builder.CreateAlloca(PassAuxiliary->Int64Type, nullptr, "size.addr");
  AllocaInst *ErrorAllocation = Builder.CreateAlloca(PassAuxiliary->Int32PtrType, nullptr, "error.addr");
  Value *ThreadIDXptr = Builder.CreateAlloca(PassAuxiliary->Int32Type, nullptr, "ThreadIDX");
  Value *ThreadIDYptr = Builder.CreateAlloca(PassAuxiliary->Int32Type, nullptr, "ThreadIDY");
  Value *ThreadIDptr = Builder.CreateAlloca(PassAuxiliary->Int32Type, nullptr, "ThreadID");
  Builder.CreateStore(DeviceA, DeviceAAllocation);
  Builder.CreateStore(DeviceB, DeviceBAllocation);
  Builder.CreateStore(ArraySize, DeviceArraySizeAllocation);
  Builder.CreateStore(Error, ErrorAllocation);

  Value *BlockID = Builder.CreateCall(PassAuxiliary->CudaDimensionFunctions[0]);
  Value *BlockDimension = Builder.CreateCall(PassAuxiliary->CudaDimensionFunctions[4]);
  Value *BlockXGrid = Builder.CreateMul(BlockDimension, BlockID,"mul");
  Value *ThreadXNumber = Builder.CreateCall(PassAuxiliary->CudaDimensionFunctions[2]);
  Value *ThreadIDX = Builder.CreateAdd(BlockXGrid, ThreadXNumber, "add");
  Builder.CreateStore(ThreadIDX, ThreadIDXptr);

  Value *BlockIDY = Builder.CreateCall(PassAuxiliary->CudaDimensionFunctions[1]);
  Value *BlockDimensionY = Builder.CreateCall(PassAuxiliary->CudaDimensionFunctions[5]);
  Value *BlockXGridY = Builder.CreateMul(BlockIDY, BlockDimensionY,"mul");
  Value *ThreadNumberY = Builder.CreateCall(PassAuxiliary->CudaDimensionFunctions[3]);
  Value *ThreadIDY = Builder.CreateAdd(BlockXGridY, ThreadNumberY, "add");
  Builder.CreateStore(ThreadIDY, ThreadIDYptr);

  Value *SecondBlockDimension = Builder.CreateCall(PassAuxiliary->CudaDimensionFunctions[4]);
  Value *ThreadIDYptrXSecondBlockDimension = Builder.CreateMul(SecondBlockDimension, Builder.CreateLoad(ThreadIDYptr),"mul");
  Value *ThreadID = Builder.CreateAdd(ThreadIDYptrXSecondBlockDimension, Builder.CreateLoad(ThreadIDXptr), "add");
  Builder.CreateStore(ThreadID, ThreadIDptr);

  Value *TID = Builder.CreateLoad(ThreadIDptr);
  Value *Extented = Builder.CreateSExt(TID, PassAuxiliary->Int64Type);
  Value *ArraySizeValue = Builder.CreateLoad(DeviceArraySizeAllocation);

  Value *SizeTIDCMP = Builder.CreateICmpSLT(Extented, ArraySizeValue);
  Builder.CreateCondBr(SizeTIDCMP, IfBlock, TailBlock);

  Builder.SetInsertPoint(IfBlock);
  Value *DeviceBPointer = Builder.CreateLoad(DeviceBAllocation);

  TID = Builder.CreateLoad(ThreadIDptr);
  Value *TID64Bit = Builder.CreateSExt(TID, PassAuxiliary->Int64Type);
  Value *PointerToTIDthElementOfDeviceB = Builder.CreateInBoundsGEP(DeviceBPointer, TID64Bit);
  Value *TIDthElementOfDeviceB = Builder.CreateLoad(PointerToTIDthElementOfDeviceB);

  Value *DeviceAPointer = Builder.CreateLoad(DeviceAAllocation);
  TID = Builder.CreateLoad(ThreadIDptr);
  TID64Bit = Builder.CreateSExt(TID, PassAuxiliary->Int64Type);
  Value *PointerToTIDthElementOfDeviceA = Builder.CreateInBoundsGEP(DeviceAPointer, TID64Bit);
  Value *TIDthElementOfDeviceA = Builder.CreateLoad(PointerToTIDthElementOfDeviceA);
  Value *DeviceADeviceBCMP;

  if( TIDthElementOfDeviceA->getType()->isFloatTy() || TIDthElementOfDeviceA->getType()->isDoubleTy())
   DeviceADeviceBCMP = Builder.CreateFCmpONE(TIDthElementOfDeviceA, TIDthElementOfDeviceB);
  else
    DeviceADeviceBCMP = Builder.CreateICmpNE(TIDthElementOfDeviceA, TIDthElementOfDeviceB);

  Builder.CreateCondBr(DeviceADeviceBCMP, SecondIfBlock, TailBlock);

  Builder.SetInsertPoint(SecondIfBlock);



  Value* ErrorPointer = Builder.CreateLoad(ErrorAllocation);
  Value *PointerToFirstthElementOfError= Builder.CreateInBoundsGEP(ErrorPointer, PassAuxiliary->Zero32Bit);
  Builder.CreateStore(PassAuxiliary->One32Bit, PointerToFirstthElementOfError);
  

  Builder.CreateBr(TailBlock);

  Builder.SetInsertPoint(TailBlock);
  
  Builder.CreateRetVoid();

  return DetectionFunction;
}





#endif