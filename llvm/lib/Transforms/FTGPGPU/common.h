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
  AllocaInst* OutputAllocation;
  CallInst* MallocInstruction;
  Type* OutputType;
  Type* DestinationType;
  StringRef Name;
  Instruction* Replications[NumberOfReplication-1];
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

  Function*  CudaGlobalRegisterFunction;
  Function*  CudaRegisterFunction;
  Function*  CudaSetupArgument;
  Function*  CudaLaunch;


  Function*  ThreadIDX;
  Function*  BlockIDX;
  Function*  BlockDimX;


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
  Value* One32Bit;
  Value* Two32Bit;
  Value* Three32Bit;
  Value* MinusOne32Bit;
  Value* CudaStreamNonBlocking;
  
  Output* OutputObject;

  Value* CudaConfiguration[8];


  Constant* Four64Bit;
  Constant* Eight64Bit;

};

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

       
inline void CudaConfigure(CallInst *FunctionCall, Value* CudaConfigurations[4]) {
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
                  if(FunctionName.contains("dim3") && DimensionCall->hasMetadata("Multiplication") == false && 
                      DimensionCall->getArgOperand(0) == DestinationAllocation){
                        errs() <<  ArgIndex%2 + 1 << "\n";
                        CudaConfigurations[ArgIndex] = DimensionCall->getArgOperand(ArgIndex%2 + 1);
                        CudaConfigurations[ArgIndex+4] = DimensionCall;
                    break;
                  }
                }
                TempInstruction = TempInstruction->getNextNode();
              }
              break;
            }
          }
          TempInstruction = TempInstruction->getNextNode();
        }
      }
    }
  }
}

inline FunctionType* getTheNewKernelFunctionType(std::vector<Value *> Args, Type* ReturnType){

  std::vector<Type * > NewKernelTypes;
  for(size_t ArgIndex = 0; ArgIndex < Args.size(); ArgIndex++){
    errs() << *Args.at(ArgIndex) << "\n";
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
        errs() << *Alloca << "\n";
        SingleOutput->DestinationType = Bitcasted->getDestTy();
        SingleOutput->OutputAllocation = Alloca;
        SingleOutput->OutputType = Alloca->getAllocatedType();
        SingleOutput->MallocInstruction = ReturnFuntion;
        SingleOutput->Name = Alloca->getName();
      }
      break;
    }
  }
  return ReturnFuntion;
}


inline void createAndAllocateVariableAndreMemCpy(IRBuilder<> Builder, Output* OutputToReplicate, Instruction& LastInstructionOfPrevBB, FunctionCallee CudaMemcpy, bool IsLoop){
  LLVMContext &Context = LastInstructionOfPrevBB.getContext();
  Type *Int32Type = Type::getInt32Ty(Context);
  Type *Int8PtrType = Type::getInt8PtrTy(Context);
  Value* Three32Bit =  ConstantInt::get(Int32Type, 3);
  AllocaInst* OutputToReplicateAllocation = OutputToReplicate->OutputAllocation;
  Value* Size = OutputToReplicate->MallocInstruction->getArgOperand(1);
  Type* OutputType = OutputToReplicate->OutputType;
  Type* DestinationType = OutputToReplicate->DestinationType;
  for(int Replication = 0; Replication < NumberOfReplication - 1; Replication++){
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

    errs() << MajorityVotingFunctionName << "\n";
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

    if(dyn_cast_or_null<PointerType>(Parameter->getType()) != nullptr) SizeParameter = 4;
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

  Builder.CreateCall(CudaLaunch, {Builder.CreateBitCast(  MajorityVotingFunction, Int8PtrType)});
  registerTheFunction(MajorityVotingFunction, PassAuxiliary);


  /*
  BasicBlock *CudaRegisterBlock = dyn_cast<BasicBlock>(CudaGlobalRegister->begin());
  Instruction *FirstInstruction = dyn_cast<Instruction>(CudaRegisterBlock->begin());
  Builder.SetInsertPoint(FirstInstruction);

  Value *FunctionName =  Builder.CreateGlobalStringPtr(MajorityVotingFunctionName);
  Builder.CreateCall( CudaRegister,
      {FirstInstruction->getOperand(0),
        Builder.CreateBitCast(MajorityVotingFunction, Int8PtrType),
        FunctionName, FunctionName, 
        PassAuxiliary->MinusOne32Bit,
        PassAuxiliary->Int8PtrNull,
        PassAuxiliary->Int8PtrNull,
        PassAuxiliary->Int8PtrNull,
        PassAuxiliary->Int8PtrNull,
        PassAuxiliary->Int32PtrNull});
  */
  return MajorityVotingFunction;
}

inline void createOrInsertMajorityVotingFunction(Module& M, Output* OutputObject, Auxiliary* PassAuxiliary){
  Type* OutputType = OutputObject->OutputType;
  std::string MajorityFunctionName = "majorityVoting" + std::to_string(OutputType->getPointerElementType()->getTypeID());
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
  Value *DeviceB = Args++;
  Value *DeviceC = Args++;
  Value *ArraySize = Args;

  AllocaInst *DeviceAAllocation = Builder.CreateAlloca(OutputPointerType);
  AllocaInst *DeviceBAllocation = Builder.CreateAlloca(OutputPointerType);
  AllocaInst *DeviceCAllocation = Builder.CreateAlloca(OutputPointerType);
  AllocaInst *DeviceArraySizeAllocation = Builder.CreateAlloca(PassAuxiliary->Int64Type);


  Value *ThreadIDptr = Builder.CreateAlloca(PassAuxiliary->Int32Type);

  Builder.CreateStore(DeviceA, DeviceAAllocation);
  Builder.CreateStore(DeviceB, DeviceBAllocation);
  Builder.CreateStore(DeviceC, DeviceCAllocation);
  Builder.CreateStore(ArraySize, DeviceArraySizeAllocation);

  Value *BlockID = Builder.CreateCall(PassAuxiliary->BlockIDX);
  Value *BlockDimension = Builder.CreateCall(PassAuxiliary->BlockDimX);
  Value *BlockXGrid = Builder.CreateMul(BlockDimension, BlockID);
  Value *ThreadNumber = Builder.CreateCall(PassAuxiliary->ThreadIDX);
  Value *ThreadID = Builder.CreateAdd(BlockXGrid, ThreadNumber);

  Builder.CreateStore(ThreadID, ThreadIDptr);
  Value *TID = Builder.CreateLoad(ThreadIDptr);
  Value *Extented = Builder.CreateZExt(TID, PassAuxiliary->Int64Type);

  Value *ArraySizeValue = Builder.CreateLoad(DeviceArraySizeAllocation);

  Value *SizeTIDCMP = Builder.CreateICmpULT(Extented, ArraySizeValue);
  Builder.CreateCondBr(SizeTIDCMP, IfBlock, TailBlock);

  Builder.SetInsertPoint(IfBlock);
  Value *DeviceBPointer = Builder.CreateLoad(DeviceBAllocation);

  TID = Builder.CreateLoad(ThreadIDptr);
  Value *TID64Bit = Builder.CreateZExt(TID, PassAuxiliary->Int64Type);
  Value *PointerToTIDthElementOfDeviceB = Builder.CreateInBoundsGEP(DeviceBPointer, TID64Bit);

  Value *TIDthElementOfDeviceB = Builder.CreateLoad(PointerToTIDthElementOfDeviceB);

  Value *DeviceCPointer = Builder.CreateLoad(DeviceCAllocation);
  TID = Builder.CreateLoad(ThreadIDptr);
  TID64Bit = Builder.CreateZExt(TID, PassAuxiliary->Int64Type);
  Value *PointerToTIDthElementOfDeviceC = Builder.CreateInBoundsGEP(DeviceCPointer, TID64Bit);
  Value *TIDthElementOfDeviceC = Builder.CreateLoad(PointerToTIDthElementOfDeviceC);
  Value *DeviceADeviceBCMP;

  if( TIDthElementOfDeviceC->getType()->isFloatTy() || TIDthElementOfDeviceB->getType()->isDoubleTy())
   DeviceADeviceBCMP = Builder.CreateFCmpOEQ(TIDthElementOfDeviceC, TIDthElementOfDeviceB);
  else
    DeviceADeviceBCMP = Builder.CreateICmpEQ(TIDthElementOfDeviceC, TIDthElementOfDeviceB);

  Builder.CreateCondBr(DeviceADeviceBCMP, SecondIfBlock, TailBlock);

  Builder.SetInsertPoint(SecondIfBlock);

  DeviceCPointer = Builder.CreateLoad(DeviceCAllocation);
  TID = Builder.CreateLoad(ThreadIDptr);
  TID64Bit = Builder.CreateZExt(TID, PassAuxiliary->Int64Type);
  PointerToTIDthElementOfDeviceC = Builder.CreateInBoundsGEP(DeviceCPointer, TID64Bit);
  TIDthElementOfDeviceC = Builder.CreateLoad(PointerToTIDthElementOfDeviceC);



  Value *DeviceAPointer = Builder.CreateLoad(DeviceAAllocation);

  TID = Builder.CreateLoad(ThreadIDptr);
  TID64Bit = Builder.CreateZExt(TID, PassAuxiliary->Int64Type);
  Value *PointerToTIDthElementOfDeviceA = Builder.CreateInBoundsGEP(DeviceAPointer, TID64Bit);
  Builder.CreateStore(TIDthElementOfDeviceC, PointerToTIDthElementOfDeviceA);

  Builder.CreateBr(TailBlock);

  Builder.SetInsertPoint(TailBlock);
  Builder.CreateRetVoid();

  return MajorityVotingFunction;
}


inline void createHostRevisited(Function* NewKernelFunction, Function* OriginalFunction, Auxiliary* PassAuxiliary){

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
  Value *OriginalBased = Args--;
  Value *SecondRedundantArg = Args--;   
  Value *FirstRedundantArg = Args--;
  Type *OutputType = FirstRedundantArg->getType();
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

  Instruction* FirstRedundantAllocation = Builder.CreateAlloca(OutputType,nullptr, "FirstRedundantArg");
  Instruction* SecondRedundantAllocation = Builder.CreateAlloca(OutputType,nullptr, "SecondRedundantArg");
  Instruction* BasedAllocation = Builder.CreateAlloca(BasedType,nullptr, "OriginalBased");

  while(dyn_cast_or_null<StoreInst>(Inst) != nullptr)
    Inst = Inst->getNextNode();

  Builder.SetInsertPoint(Inst);
  Builder.CreateStore(FirstRedundantArg,FirstRedundantAllocation);
  Builder.CreateStore(SecondRedundantArg,SecondRedundantAllocation);
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

  Value* CastedArgument = Builder.CreateBitCast(FirstRedundantAllocation, PassAuxiliary->Int8PtrType);
  Value* SetupCall = Builder.CreateCall(PassAuxiliary->CudaSetupArgument,{CastedArgument, ArgumentSize, ConstantInt::get(PassAuxiliary->Int64Type, OffsetValue)});
  Instruction* Condition = dyn_cast<Instruction>(Builder.CreateICmpEQ(SetupCall, PassAuxiliary->Zero32Bit));
  Instruction* NewBBInstruction = SplitBlockAndInsertIfThen(Condition, CudaLaunchCall, false);
  BranchInst* Branch = dyn_cast<BranchInst>(Condition->getNextNode());
  Branch->setOperand(1, LastBB);

  Builder.SetInsertPoint(NewBBInstruction);
  CastedArgument = Builder.CreateBitCast(SecondRedundantAllocation, PassAuxiliary->Int8PtrType);
  OffsetValue += ArgumentSize->getSExtValue();
  SetupCall = Builder.CreateCall(PassAuxiliary->CudaSetupArgument,{CastedArgument, ArgumentSize, ConstantInt::get(PassAuxiliary->Int64Type, OffsetValue)});
  Condition = dyn_cast<Instruction>(Builder.CreateICmpEQ(SetupCall, PassAuxiliary->Zero32Bit));
  Branch = dyn_cast<BranchInst>(Condition->getNextNode());
  NewBBInstruction = SplitBlockAndInsertIfThen(Condition, Condition->getNextNode(), false);
  Branch = dyn_cast<BranchInst>(Condition->getNextNode());
  Branch->setOperand(1, LastBB);


  Builder.SetInsertPoint(NewBBInstruction);
  CastedArgument = Builder.CreateBitCast(BasedAllocation, PassAuxiliary->Int8PtrType);
  OffsetValue += ArgumentSize->getSExtValue();
  ArgumentSize =  dyn_cast<ConstantInt>((BasedType->isPointerTy()) ? PassAuxiliary->Eight64Bit : PassAuxiliary->Four64Bit);
  SetupCall = Builder.CreateCall(PassAuxiliary->CudaSetupArgument,{CastedArgument, ArgumentSize, ConstantInt::get(PassAuxiliary->Int64Type, OffsetValue)});
  Condition = dyn_cast<Instruction>(Builder.CreateICmpEQ(SetupCall, PassAuxiliary->Zero32Bit));
  Branch = dyn_cast<BranchInst>(Condition->getNextNode());
  NewBBInstruction = SplitBlockAndInsertIfThen(Condition, Condition->getNextNode(), false);
  Branch = dyn_cast<BranchInst>(Condition->getNextNode());
  Branch->setOperand(1, LastBB);


  registerTheFunction(NewKernelFunction, PassAuxiliary);
  
}




#endif