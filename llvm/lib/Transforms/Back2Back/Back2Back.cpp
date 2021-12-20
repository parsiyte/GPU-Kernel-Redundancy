#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Argument.h"
#include "llvm/IR/Constant.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Metadata.h"
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
#include "llvm/Transforms/Utils/Local.h"
#include <algorithm>
#include <cstddef>
#include <string>
#include <system_error>
#include <utility>
#include <vector>
using namespace llvm;
#define ArgumanOrder 1 // Cuda Register Fonksiyonu çağrılırken 1 arguman fonksiyonu veriyor.
    // Gelecek Cuda versiyonlarında değişme ihtimaline karşı en üste tanımladık.
#define NumberOfReplication 3
#define STREAMENABLED false

namespace {

struct Device : public ModulePass {
  static char ID;
  Device() : ModulePass(ID) {}

  bool runOnModule(Module &M) override {
    SymbolTableList<Function> &  Functions =  M.getFunctionList();
    NamedMDNode *Annotations = M.getNamedMetadata("nvvm.annotations");
    LLVMContext &C = M.getContext();

    MDNode *N = MDNode::get(C, MDString::get(C, "kernel"));


    std::vector<Function *> ValidKernels;
    for(size_t Index = 0; Index < Annotations->getNumOperands(); Index++){
      MDNode* Singleannotation = Annotations->getOperand(Index);
      MDString* Op = dyn_cast<MDString>(Singleannotation->getOperand(1));
      if(Op->getString() == "kernel"){
        ValueAsMetadata* test = dyn_cast<ValueAsMetadata>(Singleannotation->getOperand(0));
        Function* testV = dyn_cast<Function>(test->getValue());
        ValidKernels.push_back( testV);

      }


    }
    


    FunctionType* Dims = FunctionType::get(Type::getInt32Ty(C), true);

    for (auto& Func : ValidKernels) {

      FunctionType* FuncType = Func->getFunctionType();

      unsigned int NumberOfParam = FuncType->getNumParams();
      Type* OutputType = FuncType->getParamType(NumberOfParam -1);
      PointerType* OutputPtrType = dyn_cast_or_null<PointerType>(OutputType);
      errs() << *OutputType << "\n";
      if(OutputPtrType == nullptr){
        continue;
      }
      
    
    
    
    std::string FunctionName = "majorityVoting" + std::to_string(OutputType->getPointerElementType()->getTypeID());
    MDNode *TempN = MDNode::get(C, ConstantAsMetadata::get(ConstantInt::get(llvm::Type::getInt32Ty(C), 1)));
    MDNode *Con = MDNode::concatenate(N, TempN);
      if(M.getFunction(FunctionName) != nullptr){
            continue;
      }


    FunctionCallee MajorityVotingCallee = M.getOrInsertFunction(FunctionName, Type::getVoidTy(C),
        OutputPtrType, OutputPtrType, OutputPtrType,
        Type::getInt64Ty(C)
    );


    FunctionCallee GetThread = M.getFunction("llvm.nvvm.read.ptx.sreg.tid.x");

FunctionCallee GetBlockDim =
        M.getOrInsertFunction("llvm.nvvm.read.ptx.sreg.ntid.x", GetThread.getFunctionType());
    FunctionCallee GetGridDim =
        M.getFunction("llvm.nvvm.read.ptx.sreg.ctaid.x");

    Function *MajorityVotingFunction =
        dyn_cast<Function>(MajorityVotingCallee.getCallee());
    MajorityVotingFunction->setCallingConv(CallingConv::C);

    BasicBlock *EntryBlock =
        BasicBlock::Create(C, "entry", MajorityVotingFunction);

    BasicBlock *IfBlock =
        BasicBlock::Create(C, "If", MajorityVotingFunction);
    BasicBlock *SecondIfBlock =
        BasicBlock::Create(C, "If", MajorityVotingFunction);

    BasicBlock *TailBlock =
        BasicBlock::Create(C, "Tail", MajorityVotingFunction);
    IRBuilder<> Builder(EntryBlock);

    Function::arg_iterator Args = MajorityVotingFunction->arg_begin();
    Value *DeviceA = Args++;
    Value *DeviceB = Args++;
    Value *DeviceC = Args++;
    Value *ArraySize = Args;

    AllocaInst *DeviceAAllocation = Builder.CreateAlloca(DeviceA->getType());
    AllocaInst *DeviceBAllocation = Builder.CreateAlloca(DeviceB->getType());
    AllocaInst *DeviceCAllocation = Builder.CreateAlloca(DeviceC->getType());
    AllocaInst *DeviceArraySizeAllocation =
        Builder.CreateAlloca(ArraySize->getType());

    //AllocaInst *printFAlloca = Builder.CreateAlloca(d->getScalarType());

    Value *ThreadIDptr = Builder.CreateAlloca(Type::getInt32Ty(C));

    Builder.CreateStore(DeviceA, DeviceAAllocation);
    Builder.CreateStore(DeviceB, DeviceBAllocation);
    Builder.CreateStore(DeviceC, DeviceCAllocation);
    Builder.CreateStore(ArraySize, DeviceArraySizeAllocation);


    Value *ThreadBlock = Builder.CreateCall(GetBlockDim);
    Value *ThreadGrid = Builder.CreateCall(GetGridDim);
    Value *BlockXGrid = Builder.CreateMul(ThreadBlock, ThreadGrid);
    Value *ThreadNumber = Builder.CreateCall(GetThread);
    Value *ThreadID = Builder.CreateAdd(BlockXGrid, ThreadNumber);
    Builder.CreateStore(ThreadID, ThreadIDptr);
    Value *TID = Builder.CreateLoad(ThreadIDptr);
    Value *Extented = Builder.CreateZExt(TID, Type::getInt64Ty(C));
    // errs() << *TID << "\n" << *(bitCast->getDestTy()) << "\n";
    // auto XX =  Builder.CreateInBoundsGEP(printFAlloca,
    // {ConstantInt::get(Type::getInt32Ty(Context),0),ConstantInt::get(Type::getInt32Ty(Context),0)});
    Value *ArraySizeValue = Builder.CreateLoad(DeviceArraySizeAllocation);

    // Builder.CreateStore(TID,XX);

    Value *SizeTidCMP = Builder.CreateICmpULT(Extented, ArraySizeValue);
    Builder.CreateCondBr(SizeTidCMP, IfBlock, TailBlock);

    Builder.SetInsertPoint(IfBlock);

    Value *DeviceBPointer = Builder.CreateLoad(DeviceBAllocation);



    TID = Builder.CreateLoad(ThreadIDptr);
    Value *TID64Bit = Builder.CreateZExt(TID, Type::getInt64Ty(C));
    Value *PointerToTIDthElementOfDeviceB =
        Builder.CreateInBoundsGEP(DeviceBPointer, TID64Bit);
    Value *TIDthElementOfDeviceB =
        Builder.CreateLoad(PointerToTIDthElementOfDeviceB);

    Value *DeviceCPointer = Builder.CreateLoad(DeviceCAllocation);
    TID = Builder.CreateLoad(ThreadIDptr);
    TID64Bit = Builder.CreateZExt(TID, Type::getInt64Ty(C));
    Value *PointerToTIDthElementOfDeviceC =
        Builder.CreateInBoundsGEP(DeviceCPointer, TID64Bit);
    Value *TIDthElementOfDeviceC =
        Builder.CreateLoad(PointerToTIDthElementOfDeviceC);
    Value *DeviceADeviceBCMP;
    if( TIDthElementOfDeviceC->getType()->isFloatTy() || TIDthElementOfDeviceB->getType()->isDoubleTy())
    DeviceADeviceBCMP = Builder.CreateFCmpOEQ(TIDthElementOfDeviceC, TIDthElementOfDeviceB);
    else
    DeviceADeviceBCMP = Builder.CreateICmpEQ(TIDthElementOfDeviceC, TIDthElementOfDeviceB);

    Builder.CreateCondBr(DeviceADeviceBCMP, SecondIfBlock, TailBlock);

    Builder.SetInsertPoint(SecondIfBlock);

    DeviceCPointer = Builder.CreateLoad(DeviceCAllocation);
    TID = Builder.CreateLoad(ThreadIDptr);
    TID64Bit = Builder.CreateZExt(TID, Type::getInt64Ty(C));
    PointerToTIDthElementOfDeviceC =
        Builder.CreateInBoundsGEP(DeviceCPointer, TID64Bit);
    TIDthElementOfDeviceC = Builder.CreateLoad(PointerToTIDthElementOfDeviceC);



    Value *DeviceAPointer = Builder.CreateLoad(DeviceAAllocation);
    TID = Builder.CreateLoad(ThreadIDptr);
    TID64Bit = Builder.CreateZExt(TID, Type::getInt64Ty(C));
    Value *PointerToTIDthElementOfDeviceA =
        Builder.CreateInBoundsGEP(DeviceAPointer, TID64Bit);
    Builder.CreateStore(TIDthElementOfDeviceC,
                        PointerToTIDthElementOfDeviceA);

    Builder.CreateBr(TailBlock);

    Builder.SetInsertPoint(TailBlock);

    /*
            Builder.CreateCall(
                M.getFunction("vprintf"),
                {Builder.CreateGlobalStringPtr("ConfigurationCall'nın içindeyiz\n"),
                 ConstantPointerNull::get(Type::getInt8PtrTy(M.getContext()))});
        */
    Annotations->addOperand(MDNode::concatenate(MDNode::get(C, ValueAsMetadata::get(MajorityVotingFunction)), Con));

    Builder.CreateRetVoid();


    }
  


    // MDNode* TempA = MDNode::get(C,
    // ValueAsMetadata::get(MajorityVotingFunction));
    // Con = MDNode::concatenate(Con, TempA);

    /*
    Function *a;
    Value *b;
    Value *c;
    Type *d;
    BitCastInst *bitCast;
    AllocaInst *alInst;

    for (Module::iterator F = M.begin(); F != M.end(); ++F) {
      for (Function::iterator fi = F->begin(); fi != F->end(); ++fi) {
        for (BasicBlock::iterator bi = fi->begin(); bi != fi->end(); ++bi) {
          if (CallInst *callInstruction = dyn_cast<CallInst>(bi)) {
            if (callInstruction->getCalledFunction()->getName() == "vprintf") {
              // errs() << callInstruction->getNumArgOperands() << "+++++\n";
              a = callInstruction->getCalledFunction();
              b = callInstruction->getArgOperand(0);
              c = callInstruction->getArgOperand(1);
              errs() << *callInstruction << "\n";
              bitCast = dyn_cast<BitCastInst>(c);
              alInst = dyn_cast<AllocaInst>(bitCast->getOperand(0));
              d = alInst->getAllocatedType();

              // d = ->getType();
            }
          }
          // errs() << *bi << "\n";
        }
      }
    }
    */
    
    return false;
  }

}; // end of struct Hello

struct DHost : public ModulePass {
  static char ID;
  DHost() : ModulePass(ID) {}

  bool isReplicate(CallInst *FunctionCall) {
    return FunctionCall->hasMetadata("Redundancy");
  }

  StringRef getMetadataString(MDNode *RedundancyMetadata) {
    return cast<MDString>(RedundancyMetadata->getOperand(0))->getString();
  }

  std::vector<std::string> parseData(StringRef InputOrOutput) {
    std::string InputOrOutputString = InputOrOutput.str();
    size_t Position = InputOrOutputString.find("Inputs &");
    if (Position != std::string::npos)
      InputOrOutputString.erase(Position, 8);

    std::string Variable = "";
    std::vector<std::string> Variables;
    for (size_t CharIndex = 0; CharIndex <= InputOrOutputString.size();
         CharIndex++) {
      char CurrentChar = '\0';
      if (CharIndex != InputOrOutputString.size())
        CurrentChar = InputOrOutputString.at(CharIndex);

      if ((CurrentChar == '&' || CharIndex == InputOrOutputString.size())) {
        Variables.push_back(Variable);
        Variable = "";
        continue;
      }

      Variable += CurrentChar;
    }
    return Variables;
  }

  std::pair<std::vector<std::string>, std::vector<std::string>>
  parseMetadataString(StringRef MetadataString) {
    std::pair<StringRef, StringRef> InputsAndOutputs =
        MetadataString.split("Outputs &");
    StringRef InputsAsString = InputsAndOutputs.first;
    StringRef OutputsAsString = InputsAndOutputs.second;
    std::vector<std::string> Inputs = parseData(InputsAsString);
    std::vector<std::string> Outputs = parseData(OutputsAsString);
    return std::make_pair(Inputs, Outputs);
  }

  std::pair<Value *, std::pair<Type *, Type *>>
  getSizeofDevice(std::vector<CallInst *> CudaMallocFunctionCalls,
                  std::string Output) {
    Value *Size = nullptr;
    std::pair<Type *, Type *> Types;
    for (size_t Index = 0; Index < CudaMallocFunctionCalls.size(); Index++) {
      CallInst *CudaMallocFunctionCall = CudaMallocFunctionCalls.at(Index);
      AllocaInst *AllocaVariable = nullptr;
      Type *DestinationType = nullptr;
      if (BitCastInst *BitCastVariable =
              dyn_cast<BitCastInst>(CudaMallocFunctionCall->getArgOperand(0))) {
        AllocaVariable = dyn_cast<AllocaInst>(BitCastVariable->getOperand(0));
        DestinationType = dyn_cast<PointerType>(BitCastVariable->getDestTy())
                              ->getElementType();
      } else if (dyn_cast_or_null<AllocaInst>(
                     CudaMallocFunctionCall->getArgOperand(0)) != nullptr) {
        ;
        AllocaVariable =
            dyn_cast<AllocaInst>(CudaMallocFunctionCall->getArgOperand(0));
        DestinationType =
            dyn_cast<PointerType>(AllocaVariable->getAllocatedType());
      }
      std::string VariableName = AllocaVariable->getName().str();
      if (VariableName == Output) {
        Size = CudaMallocFunctionCall->getArgOperand(1);
        Types =
            std::make_pair(AllocaVariable->getAllocatedType(), DestinationType);
        break;
      }
    }
    return std::make_pair(Size, Types);
  }

  Value *createAndAllocateVariable(Value *Callee, std::string VariableName,
                                   Value *Size, IRBuilder<> Builder,
                                   Type *VariableType, Type *DestinationType) {
                                     errs() << VariableName << "-----------\n";
    Value *NullforOutputType =
        ConstantPointerNull::get(dyn_cast<PointerType>(VariableType));
    Value *Allocated =
        Builder.CreateAlloca(VariableType, nullptr, VariableName);
    Builder.CreateBitCast(Allocated, DestinationType);
    Builder.CreateStore(NullforOutputType, Allocated);
    Value *SecondReplicationCasted =
        Builder.CreateBitCast(Allocated, DestinationType->getPointerTo());
    Builder.CreateCall(Callee, {SecondReplicationCasted, Size});
    // Builder.CreateLoad(Allocated);
    return Allocated;
  }

  std::pair<std::pair<std::vector<Value *>, std::vector<Value *>>,
            std::vector<Type *>>
  getDimensions(BasicBlock *BB) {
    Value *GridDim;
    Value *BlockDim;

    std::vector<Value *> Block;
    std::vector<Value *> Grid;
    std::vector<Type *> Types;
    std::vector<CallInst *> DimensionFunctions;
    for (BasicBlock::iterator CurrentInstruction = BB->begin();
         CurrentInstruction != BB->end(); ++CurrentInstruction) {
      // errs() << *CurrentInstruction << "\n";
      if (CallInst *FunctionCall = dyn_cast<CallInst>(CurrentInstruction)) {
        StringRef FunctionName = FunctionCall->getCalledFunction()->getName();
        if (FunctionName == "cudaConfigureCall") {
          GridDim = dyn_cast<GetElementPtrInst>(
                        dyn_cast<LoadInst>(FunctionCall->getOperand(0))
                            ->getOperand(0))
                        ->getOperand(0);
          BlockDim = dyn_cast<GetElementPtrInst>(
                         dyn_cast<LoadInst>(FunctionCall->getOperand(2))
                             ->getOperand(0))
                         ->getOperand(0);

        } else if (FunctionName.contains("_ZN4dim3C2Ejjj") == true) {
          DimensionFunctions.push_back(FunctionCall);
        }
      }
    }
    return std::make_pair(std::make_pair(Block, Grid), Types);
  }

  Instruction *createDimensions(Function *Configure, IRBuilder<> Builder,
                                std::vector<Value *> Block,
                                std::vector<Value *> Grid, Type *DimensionType,
                                Type *CoercionType, Function *DimFunction,
                                Type *SteamType) {

    MaybeAlign *Align4 = new MaybeAlign(4);
    Type *Int8Ptr = Type::getInt8PtrTy(DimFunction->getContext());
    Type *Int64Type = Type::getInt64Ty(DimFunction->getContext());
    Type *Int32Type = Type::getInt32Ty(DimFunction->getContext());
    Value *Twelve64Bit = ConstantInt::get(Int64Type, 12);
    Value *Zero32bit = ConstantInt::get(Int32Type, 0);
    Value *Three32Bit = ConstantInt::get(Int32Type, 1);
    Value *Zero64Bit = ConstantInt::get(Int64Type, 0);
    Value *NullSteam =
        ConstantPointerNull::get(dyn_cast<PointerType>(SteamType));

    Value *BlockAlloca = Builder.CreateAlloca(DimensionType, nullptr);
    Value *GridAlloca = Builder.CreateAlloca(DimensionType, nullptr);

    Value *BlockCoercionAlloca = Builder.CreateAlloca(CoercionType, nullptr);
    Value *GridCoercionAlloca = Builder.CreateAlloca(CoercionType, nullptr);

    Block.insert(Block.begin(), BlockAlloca);
    Grid.insert(Grid.begin(), GridAlloca);
    Builder.CreateCall(DimFunction, {Block});
    Builder.CreateCall(DimFunction, {Grid});

    Value *BlockCoercionBitcast =
        Builder.CreateBitCast(BlockCoercionAlloca, Int8Ptr);
    Value *BlockBitcast = Builder.CreateBitCast(BlockAlloca, Int8Ptr);
    Builder.CreateMemCpy(BlockCoercionBitcast, *Align4, BlockBitcast, *Align4,
                         Twelve64Bit);
    Value *BlockX =
        Builder.CreateInBoundsGEP(BlockCoercionAlloca, {Zero32bit, Zero32bit});
    Value *BlockXArg = Builder.CreateLoad(BlockX);
    Value *BlockY =
        Builder.CreateInBoundsGEP(BlockCoercionAlloca, {Zero32bit, Three32Bit});
    Value *BlockYArg = Builder.CreateLoad(BlockY);

    Value *GridCoercionBitcast =
        Builder.CreateBitCast(GridCoercionAlloca, Int8Ptr);
    Value *GridBitcast = Builder.CreateBitCast(GridAlloca, Int8Ptr);
    Builder.CreateMemCpy(GridCoercionBitcast, *Align4, GridBitcast, *Align4,
                         Twelve64Bit);
    Value *GridX =
        Builder.CreateInBoundsGEP(GridCoercionAlloca, {Zero32bit, Zero32bit});
    Value *GridXArg = Builder.CreateLoad(GridX);
    Value *GridY =
        Builder.CreateInBoundsGEP(GridCoercionAlloca, {Zero32bit, Three32Bit});
    Value *GridYArg = Builder.CreateLoad(GridY);
    Value *ConfigureCall =
        Builder.CreateCall(Configure, {BlockXArg, BlockYArg, GridXArg, GridYArg,
                                       Zero64Bit, NullSteam});
    Value *Condition = Builder.CreateICmpNE(ConfigureCall, Three32Bit);
    Instruction *NewInstruction = SplitBlockAndInsertIfThen(
        Condition, dyn_cast<Instruction>(Condition)->getNextNode(), false);
    return NewInstruction;
  }

  Instruction *replicateTheFunction(IRBuilder<> Builder, CallInst *FunctionCall,
                                    std::vector<Value *> CreatedOutputs,
                                    std::vector<Value *> Args, int Z) {
    std::vector<Value *> Parameters;
    for (size_t Index = 0; Index < Args.size(); Index++) {
      Instruction *Load = Builder.CreateLoad(Args.at(Index));
      Parameters.push_back(Load);
    }

    for (size_t Index = 0; Index < CreatedOutputs.size(); Index++) {
      Instruction *Load = Builder.CreateLoad(CreatedOutputs.at(Index));
      Parameters.push_back(Load);
    }
    Function *Function = FunctionCall->getCalledFunction();

    return dyn_cast<Instruction>(Builder.CreateCall(Function, Parameters));
  }

  std::vector<Value *> getArgs(CallInst *FunctionCall,
                               std::vector<std::string> Outputs) {
    std::vector<Value *> Args;
    for (size_t Index = 0; Index < FunctionCall->arg_size(); Index++) {
      LoadInst *ArgLoad =
          dyn_cast<LoadInst>(FunctionCall->getArgOperand(Index));
      AllocaInst *ArgAlloca = dyn_cast<AllocaInst>(ArgLoad->getOperand(0));
      std::string ArgName = ArgAlloca->getName().str();
      bool IsOutput =
          std::find(Outputs.begin(), Outputs.end(), ArgName) == Outputs.end();
      if (IsOutput) {
        Args.push_back(ArgAlloca);
      }
    }
    return Args;
  }

  Function *createMajorityVoting(Module &M, PointerType *MajorityVotingPointerType) {

      std::string MajorityVotingFunctionName = "majorityVoting" + std::to_string(MajorityVotingPointerType->getPointerElementType()->getTypeID());
      errs() << MajorityVotingFunctionName << "\n";
    // std::to_string(MajorityVotingPointerType->getTypeID());

    Function *CudaRegisterFunction = M.getFunction("__cuda_register_globals");
    Function *CudaRegisterFunction2 = M.getFunction("__cudaRegisterFunction");
    Function *CudaSetupArgument = M.getFunction("cudaSetupArgument");

    Function *CudaLaunch = M.getFunction("cudaLaunch");

    LLVMContext &Context = M.getContext();
    Type *Int64Type = Type::getInt64Ty(Context);
    Type *Int32Type = Type::getInt32Ty(Context);
    Value *Zero32bit = ConstantInt::get(Int32Type, 0);
    PointerType *Int8PtrType = Type::getInt8PtrTy(Context);
    PointerType *Int32PtrType = Type::getInt32PtrTy(Context);

    FunctionCallee MajorityVotingCallee = M.getOrInsertFunction(
        MajorityVotingFunctionName, Type::getVoidTy(Context),
        MajorityVotingPointerType, MajorityVotingPointerType,
        MajorityVotingPointerType, Int64Type);


    std::vector<Value *> Parameters;
    Function *MajorityVotingFunction =
        dyn_cast<Function>(MajorityVotingCallee.getCallee());
    MajorityVotingFunction->setCallingConv(CallingConv::C);
    Function::arg_iterator Args = MajorityVotingFunction->arg_begin();
    Value *A = Args++;
    A->setName("A");

    Value *B = Args++;
    B->setName("B");

    Value *C = Args++;
    C->setName("C");

    Value *Size = Args++;
    Size->setName("Size");

    BasicBlock *EntryBlock =
        BasicBlock::Create(Context, "entry", MajorityVotingFunction);

    IRBuilder<> Builder(EntryBlock);

    Builder.SetInsertPoint(EntryBlock);

    Value *Aptr =
        Builder.CreateAlloca(MajorityVotingPointerType, nullptr, "A.addr");

    Value *Bptr =
        Builder.CreateAlloca(MajorityVotingPointerType, nullptr, "B.addr");

    Value *Cptr =
        Builder.CreateAlloca(MajorityVotingPointerType, nullptr, "C.addr");

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
      if (Parameter->getType() == Int32PtrType)
        SizeParameter = 4;
      else
        SizeParameter = 8; // Diğerleri pointer olduğu için herhalde. Char*,
                           // float*, int* aynı çıktı.

      Value *SizeValue = ConstantInt::get(Int64Type, SizeParameter);
      Value *CudaSetupArgumentCall = Builder.CreateCall(
          CudaSetupArgument, {BitcastParameter, SizeValue, OffsetValue});
      Instruction *IsError = dyn_cast<Instruction>(
          Builder.CreateICmpEQ(CudaSetupArgumentCall, Zero32bit));
      if (Index == 0)
        Builder.CreateRetVoid(); // Buraya daha akıllıca çözüm bulmak gerekiyor.
                                 // Sevimli gözükmüyor.

      Instruction *SplitPoint =
          SplitBlockAndInsertIfThen(IsError, IsError->getNextNode(), false);

      SplitPoint->getParent()->setName("setup.next");

      Builder.SetInsertPoint(SplitPoint);
      Offset += SizeParameter;
    }

    Builder.CreateCall(CudaLaunch, {Builder.CreateBitCast(
                                       MajorityVotingFunction, Int8PtrType)});

    BasicBlock *CudaRegisterBlock =
        dyn_cast<BasicBlock>(CudaRegisterFunction->begin());
    Instruction *FirstInstruction =
        dyn_cast<Instruction>(CudaRegisterBlock->begin());
    Builder.SetInsertPoint(FirstInstruction);

    Value *FunctionName =
        Builder.CreateGlobalStringPtr(MajorityVotingFunctionName);
    Builder.CreateCall(
        CudaRegisterFunction2,
        {FirstInstruction->getOperand(0),
         Builder.CreateBitCast(MajorityVotingFunction, Int8PtrType),
         FunctionName, FunctionName, ConstantInt::get(Int32Type, -1),
         ConstantPointerNull::get(Int8PtrType),
         ConstantPointerNull::get(Int8PtrType),
         ConstantPointerNull::get(Int8PtrType),
         ConstantPointerNull::get(Int8PtrType),
         ConstantPointerNull::get(Int32PtrType)});
    return MajorityVotingFunction;
  }

void reMemCpy( IRBuilder<> Builder, std::string VariableName, Instruction* NewOutput, std::vector<CallInst *> CudaMemcpyFunctionCalls){
  for(size_t Index = 0; Index < CudaMemcpyFunctionCalls.size(); Index++){
    CallInst* MemCpy = CudaMemcpyFunctionCalls.at(Index);
    BitCastInst* Destination = dyn_cast<BitCastInst>(MemCpy->getArgOperand(0));
    LoadInst* LoadDestination =  dyn_cast<LoadInst>(Destination->getOperand(0));
    AllocaInst* AllocationDestination =  dyn_cast<AllocaInst>(LoadDestination->getOperand(0));
    StringRef AllocatedVariableName =  AllocationDestination->getName();
    if(AllocatedVariableName == VariableName){
      Value* LoadedNewOutput = Builder.CreateLoad(NewOutput);
      Value* BitCasted =  Builder.CreateBitCast(LoadedNewOutput,Destination->getDestTy() );
      Value* LoadedSource = Builder.CreateLoad(AllocationDestination);
      Value* BitCastedDestination = Builder.CreateBitCast(LoadedSource, Destination->getDestTy());
      errs() << *BitCastedDestination << "\n";
      
      CallInst* Cloned = dyn_cast<CallInst>(MemCpy->clone());
      Cloned->setArgOperand(0, BitCasted);
      Cloned->setArgOperand(1, BitCastedDestination);
      Cloned->setArgOperand(3, ConstantInt::get(Type::getInt32Ty(MemCpy->getContext()),3));
      Cloned->insertAfter(dyn_cast<Instruction>(BitCastedDestination));
      
      break;
    }
  }



}

bool isForLoop(BasicBlock* PrevBB){
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

Value* getTheOutput(std::vector<CallInst *> CudaMallocFunctionCalls, std::string Output) {
    AllocaInst *AllocaVariable = nullptr;
    for (size_t Index = 0; Index < CudaMallocFunctionCalls.size(); Index++) {
      CallInst *CudaMallocFunctionCall = CudaMallocFunctionCalls.at(Index);
      if (BitCastInst *BitCastVariable = dyn_cast<BitCastInst>(CudaMallocFunctionCall->getArgOperand(0))) {
        AllocaVariable = dyn_cast<AllocaInst>(BitCastVariable->getOperand(0));
      } else if (dyn_cast_or_null<AllocaInst>( CudaMallocFunctionCall->getArgOperand(0)) != nullptr) {
        AllocaVariable = dyn_cast<AllocaInst>(CudaMallocFunctionCall->getArgOperand(0));
      }

    }
    return AllocaVariable;

  }

struct Output {
  AllocaInst* OutputAllocation;
  CallInst* MallocInstruction;
  Type* OutputType;
  Type* DestinationType;
  std::string Name;
  std::vector< Instruction *> Replications;
};

void parseOutput(std::vector<CallInst *> CudaMallocFunctionCalls, Output* SingleOutput){
  std::string VariableName = SingleOutput->Name;
  AllocaInst *AllocaVariable = nullptr;
  Type *DestinationType = nullptr;
    for( size_t Index = CudaMallocFunctionCalls.size(); Index-- > 0; ){
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
void createAndAllocateVariableAndreMemCpy(IRBuilder<> Builder, Output* OutputToReplicate, Instruction& LastInstructionOfPrevBB, FunctionCallee CudaMemcpy, bool IsLoop){
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
    errs() << *CudaMemcpy.getCallee() << "\n";
    errs() << *Builder.CreateCall(CudaMemcpy, {BitcastedCloned, BitcastedOutput, Size, Three32Bit}) << "\n";
    OutputToReplicate->Replications.push_back(NewAllocated);
  }

}


  bool runOnModule(Module &M) override {
    Function *CudaMemCpy = M.getFunction("cudaMemcpy");
    StringRef StreamCreateFunctionName = "cudaStreamCreateWithFlags";
    FunctionCallee StreamCreateFunction = nullptr;
    CallInst *ConfigurationCall;
    LLVMContext &Context = M.getContext();
    Type *Int32Type = Type::getInt32Ty(Context);
    Value *Zero32bit = ConstantInt::get(Int32Type, 0);
    Value *One32Bit = ConstantInt::get(Int32Type, 1);
    Value * CudaStreamNonBlocking = One32Bit;
    Type *StreamType;
    std::vector<CallInst *> CudaMallocFunctionCalls;
    std::vector<CallInst *> CudaMemcpyFunctionCalls;
    Value *StreamArray;
    for (Module::iterator F = M.begin(); F != M.end(); ++F) {
      for (Function::iterator BB = F->begin(); BB != F->end(); ++BB) {
        for (BasicBlock::iterator CurrentInstruction = BB->begin();
             CurrentInstruction != BB->end(); ++CurrentInstruction) {
          if (CallInst *FunctionCall = dyn_cast<CallInst>(CurrentInstruction)) {
            StringRef FunctionName =  FunctionCall->getCalledFunction()->getName();
            if (isReplicate(FunctionCall)) {
              
              BasicBlock* CurrentBB = dyn_cast<BasicBlock>(BB);
              BasicBlock* NextBB = CurrentBB->getNextNode();
              BasicBlock* PrevBB = CurrentBB->getPrevNode();

              Function* FunctionToReplicate = FunctionCall->getCalledFunction();

              bool IsLoop = isForLoop(PrevBB);

              Instruction* FirstInstructionOfNextBB = NextBB->getFirstNonPHI();
              Instruction* FirstInstructionOfPrevBB = PrevBB->getFirstNonPHI();
              Instruction& LastInstructionOfPrevBB = PrevBB->back();
              IRBuilder<> Builder(FunctionCall);  
              if(STREAMENABLED == true){
                errs() << "BURADA\n";
                ArrayType *ArrayType = ArrayType::get(StreamType, NumberOfReplication);
                Builder.SetInsertPoint(FirstInstructionOfPrevBB);
                StreamArray = Builder.CreateAlloca(ArrayType, nullptr, "streams");
                Value* IthStream = Builder.CreateInBoundsGEP(StreamArray, {Zero32bit, Zero32bit}, "arrayidx"); // Bunu zaten özgün çağrıya verdiğimiz için direkt 0 verebiliriz.
                StreamCreateFunction = M.getOrInsertFunction(StreamCreateFunctionName, Int32Type, IthStream->getType(), Int32Type);
                Builder.CreateCall(StreamCreateFunction, {IthStream, CudaStreamNonBlocking});
                Value* LoadedStream = Builder.CreateLoad(IthStream);
                ConfigurationCall->setArgOperand(5, LoadedStream);
              } 
              MDNode *RedundancyMetadata = FunctionCall->getMetadata("Redundancy");
              StringRef MetadataString =  getMetadataString(RedundancyMetadata);
              std::pair<std::vector<std::string>, std::vector<std::string>> InputsAndOutputs = parseMetadataString(MetadataString);
              std::vector<std::string> Inputs = InputsAndOutputs.first;
              std::vector<std::string> Outputs = InputsAndOutputs.second;
              std::vector<Value *> CreatedOutputs;
              std::vector<Output > OutputsToBeReplicated;
            
              for (size_t Index = 0; Index < Outputs.size(); Index++) {
                  std::string VariableName = Outputs[Index];
                  Output SingleOutput;
                  SingleOutput.Name = VariableName;                  
                  parseOutput(CudaMallocFunctionCalls, &SingleOutput);
                  createAndAllocateVariableAndreMemCpy(Builder, &SingleOutput, LastInstructionOfPrevBB, CudaMemCpy, IsLoop);
                  OutputsToBeReplicated.push_back(SingleOutput);
              }
              
              for(int ReplicationIndex = 1; ReplicationIndex < NumberOfReplication; ReplicationIndex++){
                CallInst* ClonedConfigureCall = dyn_cast<CallInst>(ConfigurationCall->clone());
                ClonedConfigureCall->insertBefore(FirstInstructionOfNextBB);  
                if(STREAMENABLED == true){
                    Builder.SetInsertPoint(ClonedConfigureCall);
                    Value* IthStream = Builder.CreateInBoundsGEP(StreamArray, {Zero32bit, ConstantInt::get(Int32Type, ReplicationIndex)}, "arrayidx");
                    StreamCreateFunction = M.getOrInsertFunction(StreamCreateFunctionName, Int32Type, IthStream->getType(), Int32Type);
                    Builder.CreateCall(StreamCreateFunction, {IthStream, CudaStreamNonBlocking});
                    Value* LoadedStream = Builder.CreateLoad(IthStream);
                    ClonedConfigureCall->setArgOperand(5, LoadedStream);
                  }

                  Builder.SetInsertPoint(ClonedConfigureCall->getNextNode());
                  Instruction* ConfgurationCheck = dyn_cast<Instruction>(Builder.CreateICmpNE(ClonedConfigureCall, One32Bit));
                  Instruction* NewBasicBlockFirstInstruction = SplitBlockAndInsertIfThen(ConfgurationCheck, ConfgurationCheck->getNextNode(), false);

                  int NumberOfArg = FunctionCall->getNumArgOperands();
                  std::vector<Value *> ArgsOfReplicationFunction;
                  for(int ArgIndex = 0; ArgIndex < NumberOfArg - 1; ArgIndex++){ // Outputu çıkartıyoruz
                    Value* Arg = FunctionCall->getArgOperand(ArgIndex);
                    Value* Ref = Arg;
                    std::vector<Instruction * > InstructionToClone;
                    if(Instruction* ArgAsInstruction = dyn_cast<Instruction>(Arg)){
                      Instruction* CloneLocation = NewBasicBlockFirstInstruction;
                      while(true){
                        errs() << *ArgAsInstruction << "\n";
                        InstructionToClone.push_back(ArgAsInstruction);
                        ArgAsInstruction = dyn_cast<Instruction>(ArgAsInstruction->getOperand(0));
                        bool IfAlloca = dyn_cast_or_null<AllocaInst>(ArgAsInstruction) != nullptr;
                        if(IfAlloca)
                          break;
                      }
                      errs() << "#############\n";
                      Instruction* PrevCloned = nullptr;

                      for (unsigned Index = InstructionToClone.size(); Index-- > 1; ){
                        Instruction* Cloned = InstructionToClone.at(Index)->clone();
                        Cloned->insertBefore(CloneLocation);
                        if(PrevCloned != nullptr){
                          Cloned->setOperand(0, PrevCloned);
                        }
                         PrevCloned = Cloned;
                      } 
                      
                      errs() << "#############\n";
                      if(PrevCloned != nullptr){
                          Builder.SetInsertPoint(PrevCloned->getNextNode());
                         Ref = Builder.CreateLoad(PrevCloned);
                         }else{
                           Instruction* NewLoad = dyn_cast<Instruction>(Ref)->clone();
                           NewLoad->insertBefore(CloneLocation);
                           Ref = NewLoad;
                         }

                    }
                    ArgsOfReplicationFunction.push_back(Ref);
                  }
                  Builder.SetInsertPoint(NewBasicBlockFirstInstruction);
                  
                  for(size_t OutputIndex = 0; OutputIndex < OutputsToBeReplicated.size(); OutputIndex++){
                    Instruction* NewOutput = OutputsToBeReplicated.at(OutputIndex).Replications.at(ReplicationIndex-1);
                    ArgsOfReplicationFunction.push_back(Builder.CreateLoad(NewOutput));
                  }

                  Builder.CreateCall(FunctionToReplicate, ArgsOfReplicationFunction);
                  }


                  Instruction* CurrectInsertionPoint = dyn_cast<Instruction>(Builder.GetInsertPoint());
                  BasicBlock* CurrentBasicBlock = CurrectInsertionPoint->getParent();
                  BasicBlock* NextBasicBlock = CurrentBasicBlock->getNextNode();
                  Instruction* FirstInstrionOfNextBB = NextBasicBlock->getFirstNonPHI();

                
                for(size_t OutputIndex = 0; OutputIndex < OutputsToBeReplicated.size(); OutputIndex++ ){

                  Instruction* ClonedConfigure = ConfigurationCall->clone();
                  ClonedConfigure->insertAfter(FirstInstrionOfNextBB);
                  Builder.SetInsertPoint(ClonedConfigure->getNextNode());
                  Instruction* ConfigureCheck = dyn_cast<Instruction>(Builder.CreateICmpNE(ClonedConfigure, One32Bit));
                  Instruction* FirstInstructionOfNextBB = SplitBlockAndInsertIfThen(ConfigureCheck, ConfigureCheck->getNextNode(), false);
                  Builder.SetInsertPoint(FirstInstructionOfNextBB);


                  Output CurrentOutput = OutputsToBeReplicated.at(OutputIndex);
                  Type* OutputType = CurrentOutput.OutputType;
                  std::string MajorityFunctionName = "majorityFunction" + std::to_string(OutputType->getPointerElementType()->getTypeID());
                  Function* MajorityFunction = M.getFunction(MajorityFunctionName);
                  if(MajorityFunction == nullptr) MajorityFunction = createMajorityVoting(M, dyn_cast<PointerType>(OutputType));

                  Value* OrijinalOutput = CurrentOutput.OutputAllocation;
                  Value* FirstReplicationOutput = CurrentOutput.Replications.at(0);
                  Value* SecondReplicationOutput = CurrentOutput.Replications.at(1);
                  Value* SizeOfOutput = CurrentOutput.MallocInstruction->getArgOperand(1);

                  Value* LoadedOrijinalOutput = Builder.CreateLoad(OrijinalOutput);
                  Value* LoadedFirstReplicationOutput = Builder.CreateLoad(FirstReplicationOutput);
                  Value* LoadedSecondReplicationOutput = Builder.CreateLoad(SecondReplicationOutput);
                  //Value* LoadedSizeOfOutput = Builder.CreateLoad(SizeOfOutput);

                  Builder.CreateCall(MajorityFunction, {LoadedOrijinalOutput, LoadedFirstReplicationOutput, LoadedSecondReplicationOutput, SizeOfOutput});

                }
                

            } else if (FunctionName.contains("cudaMalloc")) {
              CudaMallocFunctionCalls.push_back(FunctionCall);
            } else if (FunctionName == "cudaConfigureCall") {
              ConfigurationCall = FunctionCall;
              StreamType = FunctionCall->getArgOperand(5)->getType();
            }else if (FunctionName == "cudaMemcpy") {
              CudaMemcpyFunctionCalls.push_back(FunctionCall);
            }
          }
        }
      }
    }
    
        return false;
  }
};

} // end of anonymous namespace

char Device::ID = -2;
char DHost ::ID = -4;


static RegisterPass<Device> XX("Back2BackDevice", "Hello World Pass", false, false);
static RegisterPass<DHost> YXXX("Back2BackHost", "Hello World Pass",
                                 false /* Only looks at CFG */,
                                 false /* Analysis Pass */);

static RegisterStandardPasses YX(PassManagerBuilder::EP_EarlyAsPossible,
                                 [](const PassManagerBuilder &Builder,
                                    legacy::PassManagerBase &PM) {
                                   PM.add(new Device());
                                 });

static RegisterStandardPasses YYX(PassManagerBuilder::EP_EarlyAsPossible,
                                  [](const PassManagerBuilder &Builder,
                                     legacy::PassManagerBase &PM) {
                                    PM.add(new DHost());
                                  });
