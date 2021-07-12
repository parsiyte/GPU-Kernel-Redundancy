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
#include <utility>
#include <vector>
#include "llvm/Transforms/Utils/Cloning.h"
using namespace llvm;
#define Replication 3
#define NumberofDimension 2

namespace {
struct RMTDevice : public ModulePass {
  static char ID;
  RMTDevice() : ModulePass(ID) {}


  StoreInst* getTheXID(Function* Kernel){
      for (Function::iterator BB = Kernel->begin(); BB != Kernel->end(); ++BB) {
        for (BasicBlock::iterator CurrentInstruction = BB->begin();CurrentInstruction != BB->end(); ++CurrentInstruction) {   
          if (CallInst *FunctionCall = dyn_cast<CallInst>(CurrentInstruction)) {
            StringRef FunctionName = FunctionCall->getCalledFunction()->getName();
            if(FunctionName == "llvm.nvvm.read.ptx.sreg.tid.x"){
              for (auto& U : FunctionCall->uses()) {
                  errs() << *dyn_cast<Instruction> (U.getUser()) << "\n";
                  return  dyn_cast<StoreInst>(dyn_cast<Instruction> (U.getUser())->getNextNode());  
                }
              }
        }
      }
      }
    return nullptr;
  }  
  StoreInst* getTheLastStore(Function* Kernel){
    StoreInst *LastStoreInst;
      for (Function::iterator BB = Kernel->begin(); BB != Kernel->end(); ++BB) {
        for (BasicBlock::iterator CurrentInstruction = BB->begin();CurrentInstruction != BB->end(); ++CurrentInstruction) {   
          if ( StoreInst *Store = dyn_cast<StoreInst>(CurrentInstruction)) {
           LastStoreInst = Store;
          
          } 
      }
      }
      return LastStoreInst;
  }
  StoreInst* getTheSecondLastStore(Function* Kernel){
    StoreInst *LastSecondStoreInst;
    StoreInst *LastStoreInst;
      for (Function::iterator BB = Kernel->begin(); BB != Kernel->end(); ++BB) {
        for (BasicBlock::iterator CurrentInstruction = BB->begin();CurrentInstruction != BB->end(); ++CurrentInstruction) {   
          if ( StoreInst *Store = dyn_cast<StoreInst>(CurrentInstruction)) {
            LastSecondStoreInst = LastStoreInst;
           LastStoreInst = Store;
          
          } 
      }
      }
      return LastSecondStoreInst;
  }
  Function* createMajorityFuncton(Module &M, FunctionCallee MajorityVotingCallee){


    NamedMDNode *Annotations = M.getNamedMetadata("nvvm.annotations");
    LLVMContext &Context = M.getContext();


    MDNode *N = MDNode::get(Context, MDString::get(Context, "kernel"));
    MDNode *TempN = MDNode::get(Context, ConstantAsMetadata::get(ConstantInt::get(
                                       llvm::Type::getInt32Ty(Context), 1)));
    // MDNode* TempA = MDNode::get(C,
    // ValueAsMetadata::get(MajorityVotingFunction));
    MDNode *Con = MDNode::concatenate(N, TempN);
    // Con = MDNode::concatenate(Con, TempA);

    FunctionCallee GetBlockDim =
        M.getFunction("llvm.nvvm.read.ptx.sreg.ntid.x");
    FunctionCallee GetGridDim =
        M.getFunction("llvm.nvvm.read.ptx.sreg.ctaid.x");
    FunctionCallee GetThread = M.getFunction("llvm.nvvm.read.ptx.sreg.tid.x");

    Function *MajorityVotingFunction =
        dyn_cast<Function>(MajorityVotingCallee.getCallee());
    MajorityVotingFunction->setCallingConv(CallingConv::C);



    BasicBlock *EntryBlock =
        BasicBlock::Create(Context, "entry", MajorityVotingFunction);

    BasicBlock *IfBlock =
        BasicBlock::Create(Context, "If", MajorityVotingFunction);
    BasicBlock *SecondIfBlock =
        BasicBlock::Create(Context, "If", MajorityVotingFunction);
    BasicBlock *ElseBlock =
        BasicBlock::Create(Context, "Else", MajorityVotingFunction);

    BasicBlock *TailBlock =
        BasicBlock::Create(Context, "Tail", MajorityVotingFunction);
    IRBuilder<> Builder(EntryBlock);

    Function::arg_iterator Args = MajorityVotingFunction->arg_begin();
    Value *DeviceA = Args++;
    Value *DeviceB = Args++;
    Value *DeviceC = Args++;
    Value *DeviceOutput = Args++;
    Value *ArraySize = Args;

    AllocaInst *DeviceAAllocation = Builder.CreateAlloca(DeviceA->getType());
    AllocaInst *DeviceBAllocation = Builder.CreateAlloca(DeviceB->getType());
    AllocaInst *DeviceCAllocation = Builder.CreateAlloca(DeviceC->getType());
    AllocaInst *DeviceOutputAllocation =
        Builder.CreateAlloca(DeviceOutput->getType());
    AllocaInst *DeviceArraySizeAllocation =
        Builder.CreateAlloca(ArraySize->getType());
    
    Value *ThreadIDptr = Builder.CreateAlloca(Type::getInt32Ty(Context));

    StoreInst *DeviceAStore = Builder.CreateStore(DeviceA, DeviceAAllocation);
    StoreInst *DeviceBStore = Builder.CreateStore(DeviceB, DeviceBAllocation);
    StoreInst *DeviceCStore = Builder.CreateStore(DeviceC, DeviceCAllocation);
    StoreInst *DeviceOutputStore =
        Builder.CreateStore(DeviceOutput, DeviceOutputAllocation);
    StoreInst *DeviceArraySizeStore =
        Builder.CreateStore(ArraySize, DeviceArraySizeAllocation);

    Value *ThreadBlock = Builder.CreateCall(GetBlockDim);
    Value *ThreadGrid = Builder.CreateCall(GetGridDim);
    Value *BlockXGrid = Builder.CreateMul(ThreadBlock, ThreadGrid);
    Value *ThreadNumber = Builder.CreateCall(GetThread);
    Value *ThreadID = Builder.CreateAdd(BlockXGrid, ThreadNumber);
    Builder.CreateStore(ThreadID, ThreadIDptr);
    Value *TID = Builder.CreateLoad(ThreadIDptr);
    Value *Extented = Builder.CreateZExt(TID, Type::getInt64Ty(Context));

     Value *ArraySizeValue = Builder.CreateLoad(DeviceArraySizeAllocation);

    // Builder.CreateStore(TID,XX);

    Value *SizeTidCMP = Builder.CreateICmpULT(Extented, ArraySizeValue);
    Instruction *Branch = Builder.CreateCondBr(SizeTidCMP, IfBlock, TailBlock);

    Builder.SetInsertPoint(IfBlock);

    Value *DeviceAPointer = Builder.CreateLoad(DeviceAAllocation);
    TID = Builder.CreateLoad(ThreadIDptr);
    Value *TID64Bit = Builder.CreateZExt(TID, Type::getInt64Ty(Context));
    Value *PointerToTIDthElementOfDeviceA =
        Builder.CreateInBoundsGEP(DeviceAPointer, TID64Bit);
    Value *TIDthElementOfDeviceA =
        Builder.CreateLoad(PointerToTIDthElementOfDeviceA);

    Value *DeviceBPointer = Builder.CreateLoad(DeviceBAllocation);
    TID = Builder.CreateLoad(ThreadIDptr);
    TID64Bit = Builder.CreateZExt(TID, Type::getInt64Ty(Context));
    Value *PointerToTIDthElementOfDeviceB =
        Builder.CreateInBoundsGEP(DeviceBPointer, TID64Bit);
    Value *TIDthElementOfDeviceB =
        Builder.CreateLoad(PointerToTIDthElementOfDeviceB);

    Value *DeviceADeviceBCMP =
        Builder.CreateFCmpOEQ(TIDthElementOfDeviceA, TIDthElementOfDeviceB);
    Builder.CreateCondBr(DeviceADeviceBCMP, SecondIfBlock, ElseBlock);

    Builder.SetInsertPoint(SecondIfBlock);

    DeviceAPointer = Builder.CreateLoad(DeviceAAllocation);
    TID = Builder.CreateLoad(ThreadIDptr);
    TID64Bit = Builder.CreateZExt(TID, Type::getInt64Ty(Context));
    PointerToTIDthElementOfDeviceA =
        Builder.CreateInBoundsGEP(DeviceAPointer, TID64Bit);
    TIDthElementOfDeviceA = Builder.CreateLoad(PointerToTIDthElementOfDeviceA);
    Value *DeviceOutputPointer = Builder.CreateLoad(DeviceOutputAllocation);
    TID = Builder.CreateLoad(ThreadIDptr);
    TID64Bit = Builder.CreateZExt(TID, Type::getInt64Ty(Context));
    Value *PointerToTIDthElementOfDeviceOutput =
        Builder.CreateInBoundsGEP(DeviceOutputPointer, TID64Bit);
    Builder.CreateStore(TIDthElementOfDeviceA,
                        PointerToTIDthElementOfDeviceOutput);

    Builder.CreateBr(TailBlock);

    Builder.SetInsertPoint(ElseBlock); // Burası Else
    Value *DeviceCPointer = Builder.CreateLoad(DeviceCAllocation);
    TID = Builder.CreateLoad(ThreadIDptr);
    TID64Bit = Builder.CreateZExt(TID, Type::getInt64Ty(Context));
    Value *PointerToTIDthElementOfDeviceC =
        Builder.CreateInBoundsGEP(DeviceCPointer, TID64Bit);
    Value *TIDthElementOfDeviceC =
        Builder.CreateLoad(PointerToTIDthElementOfDeviceC);
    DeviceOutputPointer = Builder.CreateLoad(DeviceOutputAllocation);
    TID = Builder.CreateLoad(ThreadIDptr);
    TID64Bit = Builder.CreateZExt(TID, Type::getInt64Ty(Context));
    PointerToTIDthElementOfDeviceOutput =
        Builder.CreateInBoundsGEP(DeviceOutputPointer, TID64Bit);
    Builder.CreateStore(TIDthElementOfDeviceC,
                        PointerToTIDthElementOfDeviceOutput);
    Builder.CreateBr(TailBlock);

    Builder.SetInsertPoint(TailBlock);

    Annotations->addOperand(MDNode::concatenate(
        MDNode::get(Context, ValueAsMetadata::get(MajorityVotingFunction)), Con));

    Builder.CreateRetVoid();
    return MajorityVotingFunction;
  }

  bool runOnModule(Module &M) override {
    
     NamedMDNode *Annotations = M.getNamedMetadata("nvvm.annotations");
     LLVMContext& Context = M.getContext();
     Type* FloatPointerType = Type::getFloatPtrTy(Context);
     Type* Int32Type = Type::getInt32Ty(Context);
     Type* Int8ptrType = Type::getInt8PtrTy(Context);
     Type* VoidType = Type::getVoidTy(Context);
    Value *Zero32Bit = ConstantInt::get(Int32Type, 0);
    Value *One32Bit = ConstantInt::get(Int32Type, 1);
    Value *Two32Bit = ConstantInt::get(Int32Type, 2);



     for(unsigned int Index = 0; Index < Annotations->getNumOperands(); Index++){
     MDNode* Operand = Annotations->getOperand(Index);
     Metadata* Feature = Operand->getOperand(1).get();
     if(cast<MDString>(Feature)->getString()  == "kernel"){
        errs() <<  *Operand->getOperand(0).get()<< "\n";
        Metadata* FunctionMetadata = cast<Metadata>(Operand->getOperand(0));
        ValueAsMetadata* AsValue = cast<ValueAsMetadata>(FunctionMetadata);
        Function* Kernel =  dyn_cast<Function>(AsValue->getValue());
        errs() << Kernel->getName() << "\n";
        if(Kernel->getName().contains("Revisited") || Kernel->getName().contains("majorityVoting15")){
          continue;
        }
        FunctionType* KernelType = Kernel->getFunctionType();
        unsigned int ParamSize = KernelType->getNumParams();
        Type* OutputType = KernelType->getParamType(ParamSize-1); // Son Parametrenin Output olduğu kabulü
        std::vector<Type *> NewFunctionType;
        for(unsigned int SIndex = 0; SIndex < ParamSize; SIndex++ ){
          NewFunctionType.push_back(KernelType->getParamType(SIndex));
        }
        for(int OutputIndex = 0; OutputIndex < Replication - 1; OutputIndex++){
          NewFunctionType.push_back(OutputType);
        }
        for(int Dimension = 0; Dimension < NumberofDimension; Dimension++ ){
          NewFunctionType.push_back(Int32Type);
        }
        FunctionType* NewKernelType = FunctionType::get(VoidType,NewFunctionType, true);
        std::string NewKernelName = Kernel->getName().str() + "Revisited";
        errs() << NewKernelName << "\n";
        FunctionCallee NewKernelAsCallee=  M.getOrInsertFunction(NewKernelName, NewKernelType);
        Function* NewKernelFunction = cast<Function>(NewKernelAsCallee.getCallee());
        
        ValueToValueMapTy VMap;
         SmallVector<ReturnInst*, 8> Returns;
          Function::arg_iterator DestI = NewKernelFunction->arg_begin();
          for (const Argument & I : Kernel->args())
            if (VMap.count(&I) == 0) {    
              DestI->setName(I.getName()); 
              VMap[&I] = &*DestI++;       
            }
 
        CloneFunctionInto(NewKernelFunction,Kernel,VMap,false,Returns);   

       Function::arg_iterator Args = NewKernelFunction->arg_end();
        Args--;
        Value *OriginalY = Args--;
        Value *OriginalX = Args--;
        Value *DA2 = Args--;
        Value *DA1 = Args--;
        
        
        DA1->setName("d_A1");
        DA2->setName("d_A2");
        OriginalX->setName("OriginalX");
        OriginalY->setName("OriginalY");
        //errs() << *NewKernelFunction << "\n";
        StoreInst* XID = getTheXID(NewKernelFunction);
        IRBuilder<> Builder(XID->getNextNode());
        Value* OriginalXAddr = Builder.CreateAlloca(Int32Type,nullptr, "OriginalX.addr");
        Builder.CreateStore(OriginalX, OriginalXAddr);
        Value* DA1Addr = Builder.CreateAlloca(FloatPointerType,nullptr, "d_A1addr");
        Builder.CreateStore(DA1, DA1Addr);
        Value* DA2Addr = Builder.CreateAlloca(FloatPointerType,nullptr, "d_A2addr");
        Builder.CreateStore(DA2, DA2Addr);
        Value* IntA = Builder.CreateAlloca(Int32Type,nullptr, "A");
        Value* ResultA = Builder.CreateUDiv(
            Builder.CreateLoad(XID->getPointerOperand()),
            Builder.CreateLoad(OriginalXAddr)
         );
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
      auto yy =  Builder.CreateStore(ResultA, IntA);
    AllocaInst *printFAlloca = Builder.CreateAlloca(d->getScalarType());
     auto XX =  Builder.CreateInBoundsGEP(printFAlloca, {ConstantInt::get(Type::getInt32Ty(Context),0),ConstantInt::get(Type::getInt32Ty(Context),0)});
     Builder.CreateStore( Builder.CreateLoad(yy->getPointerOperand()), XX);

        Builder.CreateCall(
            M.getFunction("vprintf"),
            {Builder.CreateGlobalStringPtr("%d \n"),
             Builder.CreateBitCast(printFAlloca, Int8ptrType)
             });
             
        Value* ThreadIDX = XID->getPointerOperand();
        Value* Urem = Builder.CreateURem(
            Builder.CreateLoad(XID->getPointerOperand()), 
            Builder.CreateLoad(OriginalXAddr)
        );
        
        errs() << *Builder.CreateStore(Urem, ThreadIDX ) << "\n";

        StoreInst* LastStore = getTheLastStore(NewKernelFunction);
        Value* CalculatedValue = LastStore->getValueOperand();
        GetElementPtrInst* Index = dyn_cast_or_null<GetElementPtrInst>(LastStore->getPointerOperand());
        if(Index == nullptr){
          LastStore = getTheSecondLastStore(NewKernelFunction);
          CalculatedValue = LastStore->getValueOperand();
          Index = dyn_cast_or_null<GetElementPtrInst>(LastStore->getPointerOperand());
        }
        Value* Pointer = dyn_cast<Value>(Index->idx_begin());
        errs() << *Pointer<< "Pointer\n";
        errs() << *Index<< "Index\n";

        Builder.SetInsertPoint(LastStore);
        Instruction* IsOriginal = dyn_cast<Instruction>(Builder.CreateICmpEQ(ResultA, Zero32Bit));
        Instruction* Branch = SplitBlockAndInsertIfThen(IsOriginal, IsOriginal->getNextNode(), false);
        LastStore->moveBefore(Branch);

        Builder.SetInsertPoint(dyn_cast<Instruction>(Branch->getParent()->getNextNode()->begin()));
        Instruction* IsFirstOne = dyn_cast<Instruction>(Builder.CreateICmpEQ(ResultA, One32Bit));
        Branch = SplitBlockAndInsertIfThen(IsFirstOne, IsFirstOne->getNextNode(), false);
        Builder.SetInsertPoint(dyn_cast<Instruction>(Branch));
        Value* LoadedDA1Addr = Builder.CreateLoad(DA1Addr);
        Value* DA1Location = Builder.CreateInBoundsGEP(LoadedDA1Addr,{Pointer});
        errs() << *DA1Location << "\n";
        Builder.CreateStore(CalculatedValue, DA1Location);


        errs() << *Branch << "\n";
        Builder.SetInsertPoint(dyn_cast<Instruction>(Branch->getParent()->getNextNode()->begin()));
        Instruction* IsSecondOne = dyn_cast<Instruction>(Builder.CreateICmpEQ(ResultA, Two32Bit));
        Branch = SplitBlockAndInsertIfThen(IsSecondOne, IsSecondOne->getNextNode(), false);
        Builder.SetInsertPoint(dyn_cast<Instruction>(Branch));
        Value* LoadedDA2Addr = Builder.CreateLoad(DA2Addr);
        Value* DA2Location = Builder.CreateInBoundsGEP(LoadedDA2Addr,{Pointer});
        Builder.CreateStore(CalculatedValue, DA2Location);

        MDNode *N = MDNode::get(Context, MDString::get(Context, "kernel"));
        MDNode *TempN = MDNode::get(Context, ConstantAsMetadata::get(ConstantInt::get(
                                          Int32Type, 1)));
        MDNode *Con = MDNode::concatenate(N, TempN);

      Annotations->addOperand(MDNode::concatenate(
          MDNode::get(Context, ValueAsMetadata::get(NewKernelFunction)), Con));


      FunctionCallee MajorityVotingCallee = M.getOrInsertFunction(
          "majorityVoting15", Type::getVoidTy(Context),
          Type::getFloatPtrTy(Context), Type::getFloatPtrTy(Context),
          Type::getFloatPtrTy(Context), Type::getFloatPtrTy(Context),
          Type::getInt64Ty(Context)

      );
      createMajorityFuncton(M,MajorityVotingCallee );
        }
     }
     
    return false;
  }

}; 

struct RMTHost : public ModulePass {
  static char ID;
  RMTHost() : ModulePass(ID) {}

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

  Function *createRevisted(Module &M,std::string FunctionName,FunctionCallee RevistedFunctionCallee) {
    // std::to_string(RevistedPointerType->getTypeID());

    Function *CudaRegisterFunction = M.getFunction("__cuda_register_globals");
    Function *CudaRegisterFunction2 = M.getFunction("__cudaRegisterFunction");
    Function *CudaSetupArgument = M.getFunction("cudaSetupArgument");

    Function *CudaLaunch = M.getFunction("cudaLaunch");

    LLVMContext &Context = M.getContext();
    Type *Int64Type = Type::getInt64Ty(Context);
    Type *Int32Type = Type::getInt32Ty(Context);
    Value *Zero32Bit = ConstantInt::get(Int32Type, 0);
    PointerType *Int8PtrType = Type::getInt8PtrTy(Context);
    PointerType *Int32PtrType = Type::getInt32PtrTy(Context);

    std::vector<Value *> Parameters;

    Function *RevistedFunction =
        dyn_cast<Function>(RevistedFunctionCallee.getCallee());
    RevistedFunction->setCallingConv(CallingConv::C);
    Function::arg_iterator Args = RevistedFunction->arg_begin();
    /*
    Value *OriginalY = Args--;
    OriginalY->setName("OriginalY");

    Value *OriginalX = Args--;
    OriginalX->setName("OriginalX");

    Value *DeviceA2 = Args--;
    DeviceA2->setName("DeviceA2");

    Value *DeviceA1 = Args--;
    DeviceA1->setName("DeviceA1");
*/
     BasicBlock *EntryBlock =
        BasicBlock::Create(Context, "entry", RevistedFunction);

    IRBuilder<> Builder(EntryBlock);

    Builder.SetInsertPoint(EntryBlock);
    int Offset = 0;
    int SizeParameter = 8;
    while(Args != RevistedFunction->arg_end()){
      Value* Arguman = Args++;
      Type* ArgumanType = Arguman->getType();
      if (ArgumanType == Int32PtrType || ArgumanType == Int32Type)
        SizeParameter = 4;
      else
        SizeParameter = 8; // Diğerleri pointer olduğu için herhalde. Char*,
                           // float*, int* aynı çıktı.
      MaybeAlign ArgumanAlign = MaybeAlign(SizeParameter);
      AllocaInst* ArgumanPointer = Builder.CreateAlloca(ArgumanType, nullptr, "Arguman.addr");
      ArgumanPointer->setAlignment(ArgumanAlign);
      Builder.CreateStore(Arguman, ArgumanPointer);
      Parameters.push_back(ArgumanPointer);
      Value *BitcastParameter = Builder.CreateBitCast(ArgumanPointer, Int8PtrType);
      while(Offset%SizeParameter != 0)
        Offset += 1;
      Value *OffsetValue = ConstantInt::get(Int64Type, Offset);
      errs() << *ArgumanType << "\n";
        errs() << SizeParameter << "\n";

       Value *SizeValue = ConstantInt::get(Int64Type, SizeParameter);
      Value *CudaSetupArgumentCall = Builder.CreateCall(
          CudaSetupArgument, {BitcastParameter, SizeValue, OffsetValue});
      Instruction *IsError = dyn_cast<Instruction>(
          Builder.CreateICmpEQ(CudaSetupArgumentCall, Zero32Bit));
      if (Offset == 0)
        Builder.CreateRetVoid(); // Buraya daha akıllıca çözüm bulmak gerekiyor.
                                 // Sevimli gözükmüyor.

      Instruction *SplitPoint =
          SplitBlockAndInsertIfThen(IsError, IsError->getNextNode(), false);

      SplitPoint->getParent()->setName("setup.next");

      Builder.SetInsertPoint(SplitPoint);
      Offset += SizeParameter;
    }

    Builder.CreateCall(CudaLaunch, {Builder.CreateBitCast(
                                       RevistedFunction, Int8PtrType)});

    BasicBlock *CudaRegisterBlock =
        dyn_cast<BasicBlock>(CudaRegisterFunction->begin());
    Instruction *FirstInstruction =
        dyn_cast<Instruction>(CudaRegisterBlock->begin());
    Builder.SetInsertPoint(FirstInstruction);

    Value *FunctionNameString =
        Builder.CreateGlobalStringPtr(FunctionName);
    Builder.CreateCall(
        CudaRegisterFunction2,
        {FirstInstruction->getOperand(0),
         Builder.CreateBitCast(RevistedFunction, Int8PtrType),
         FunctionNameString, FunctionNameString, ConstantInt::get(Int32Type, -1),
         ConstantPointerNull::get(Int8PtrType),
         ConstantPointerNull::get(Int8PtrType),
         ConstantPointerNull::get(Int8PtrType),
         ConstantPointerNull::get(Int8PtrType),
         ConstantPointerNull::get(Int32PtrType)});
    /*


         */
    return RevistedFunction;
  }


  Function *createMajorityVoting(Module &M,
                                 PointerType *MajorityVotingPointerType) {
    std::string MajorityVotingFunctionName = "majorityVoting15";
    // std::to_string(MajorityVotingPointerType->getTypeID());

    Function *CudaRegisterFunction = M.getFunction("__cuda_register_globals");
    Function *CudaRegisterFunction2 = M.getFunction("__cudaRegisterFunction");
    Function *CudaSetupArgument = M.getFunction("cudaSetupArgument");

    Function *CudaLaunch = M.getFunction("cudaLaunch");

    LLVMContext &Context = M.getContext();
    Type *Int64Type = Type::getInt64Ty(Context);
    Type *Int32Type = Type::getInt32Ty(Context);
    Value *Zero32Bit = ConstantInt::get(Int32Type, 0);
    PointerType *Int8PtrType = Type::getInt8PtrTy(Context);
    PointerType *Int32PtrType = Type::getInt32PtrTy(Context);

    FunctionCallee MajorityVotingCallee = M.getOrInsertFunction(
        MajorityVotingFunctionName, Type::getVoidTy(Context),
        MajorityVotingPointerType, MajorityVotingPointerType,
        MajorityVotingPointerType, MajorityVotingPointerType,
        Type::getInt64Ty(Context));
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

    Value *Output = Args++;
    Output->setName("Output");

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

    Value *Outputptr =
        Builder.CreateAlloca(MajorityVotingPointerType, nullptr, "output.addr");

    Value *Sizeptr = Builder.CreateAlloca(Int64Type, nullptr, "size.addr");

    StoreInst *StoreA = Builder.CreateStore(A, Aptr);

    Value *StoreB = Builder.CreateStore(B, Bptr);

    Value *StoreC = Builder.CreateStore(C, Cptr);

    Value *StoreOutput = Builder.CreateStore(Output, Outputptr);

    Value *StoreSize = Builder.CreateStore(Size, Sizeptr);

    Parameters.push_back(Aptr);
    Parameters.push_back(Bptr);
    Parameters.push_back(Cptr);
    Parameters.push_back(Outputptr);
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
          Builder.CreateICmpEQ(CudaSetupArgumentCall, Zero32Bit));
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

  bool runOnModule(Module &M) override {

    CallInst* Cuda;
    Function *CudaMalloc = M.getFunction("cudaMalloc");
    LLVMContext& Context = M.getContext();
    Type* Int32Type = Type::getInt32Ty(Context);
    Type* Int64Type = Type::getInt64Ty(Context);
    Type* VoidType = Type::getVoidTy(Context);
    std::vector<CallInst *> CudaMallocFunctionCalls;
      Value *One32Bit = ConstantInt::get(Int32Type, 1);
      Value *Zero32Bit = ConstantInt::get(Int32Type, 0);

    Function *ConfigureFunction = M.getFunction("cudaConfigureCall");
    
    std::vector<Value *> CreatedOutputs; // GERI TAŞI
    for (Module::iterator F = M.begin(); F != M.end(); ++F) {
      for (Function::iterator BB = F->begin(); BB != F->end(); ++BB) {
        for (BasicBlock::iterator CurrentInstruction = BB->begin(); CurrentInstruction != BB->end(); ++CurrentInstruction) {  
          if (CallInst *FunctionCall = dyn_cast<CallInst>(CurrentInstruction)) {
               StringRef FunctionName = FunctionCall->getCalledFunction()->getName();
                if(isReplicate(FunctionCall)){
                  
                   CreatedOutputs.clear();
                    FunctionType* KernelType = FunctionCall->getFunctionType();
                    unsigned int ParamSize = KernelType->getNumParams();
                    Type* OutputType = KernelType->getParamType(ParamSize-1); // Son Parametrenin Output olduğu kabulü
                    std::vector<Type *> NewFunctionType;
                    for(unsigned int SIndex = 0; SIndex < ParamSize; SIndex++ ){
                      NewFunctionType.push_back(KernelType->getParamType(SIndex));
                    }
                    for(int OutputIndex = 0; OutputIndex < Replication - 1; OutputIndex++){
                      NewFunctionType.push_back(OutputType);
                    }
                    for(int Dimension = 0; Dimension < NumberofDimension; Dimension++ ){
                      NewFunctionType.push_back(Int32Type);
                    }
                    FunctionType* NewKernelType = FunctionType::get(VoidType,NewFunctionType, true);
                    std::string NewKernelName = FunctionName.str() + "Revisited";
                    FunctionCallee NewKernelAsCallee =  M.getOrInsertFunction(NewKernelName, NewKernelType);
                    
                    IRBuilder<> Builder(dyn_cast<Instruction>(BB->getPrevNode()->begin()));
                   std::pair<Value *, std::pair<Type *, Type *>> SizeOfTheOutput;
                    MDNode *RedundancyMetadata =
                    FunctionCall->getMetadata("Redundancy");
                    StringRef MetadataString =
                        getMetadataString(RedundancyMetadata);
                    std::pair<std::vector<std::string>, std::vector<std::string>>
                        InputsAndOutputs = parseMetadataString(MetadataString);
                    std::vector<std::string> Inputs = InputsAndOutputs.first;
                      std::vector<std::string> Outputs = InputsAndOutputs.second;
                        Builder.SetInsertPoint(FunctionCall);
                        for(int i = 0; i < FunctionCall->getNumArgOperands(); i++){
                          LoadInst* LoadedArg =  dyn_cast<LoadInst>(FunctionCall->getArgOperand(i));
                          CreatedOutputs.push_back(LoadedArg);
                        }
                    Builder.SetInsertPoint(Cuda->getPrevNode());
                    for(int i = 0; i < 2; i++)
                    for (size_t Index = 0; Index < Outputs.size(); Index++) {
                    std::string VariableName = Outputs[Index];
                    SizeOfTheOutput =
                        getSizeofDevice(CudaMallocFunctionCalls, VariableName);
                    Value *NewOutput = createAndAllocateVariable(
                        CudaMalloc, VariableName, SizeOfTheOutput.first,
                        Builder, SizeOfTheOutput.second.first,
                        SizeOfTheOutput.second.second);
                     CreatedOutputs.push_back(NewOutput);
                     } 
                    // FIX ME!!!!!!!
                    CreatedOutputs.push_back(ConstantInt::get(Int32Type, 32));
                    CreatedOutputs.push_back(ConstantInt::get(Int32Type, 1  ));
                     
                    for(int I =  CreatedOutputs.size() - 4; I < CreatedOutputs.size() - 2 ; I++){
                      errs() <<  *CreatedOutputs.at(I) << "\n";
                      CreatedOutputs.at(I) = Builder.CreateLoad(CreatedOutputs.at(I));
                    }
                    //  errs() << *F << "\n";



                    Builder.SetInsertPoint(FunctionCall);
                   Instruction* NewFunctionCall = Builder.CreateCall(NewKernelAsCallee, CreatedOutputs);
                   /*
                    errs() << *FunctionCall->getArgOperand(ParamSize-1) << "\n";
                    errs() << *CreatedOutputs.at(2) << "\n";
                    errs() << *CreatedOutputs.at(3) << "\n";
                    errs() << *M.getFunction("_Z6kernelPfS_S_") << "\n";
                    Builder.CreateCall(M.getFunction("_Z6kernelPfS_S_"), {FunctionCall->getArgOperand(ParamSize-1), CreatedOutputs.at(2) , CreatedOutputs.at(3)});*/
                  
                   CurrentInstruction++;
                   FunctionCall->eraseFromParent();
                   //errs() << *CurrentInstruction << "\n";
                    createRevisted(M,NewKernelName,NewKernelAsCallee);

                    BasicBlock *CurrentBB = NewFunctionCall->getParent();
                    BasicBlock *NextBB = CurrentBB->getNextNode();
                    BasicBlock *PrevBB = CurrentBB->getPrevNode();
                    Instruction *FirstInstruction = dyn_cast<Instruction>(PrevBB->begin());
                    FirstInstruction = dyn_cast<Instruction>(NextBB->begin());


                    Builder.SetInsertPoint(FirstInstruction);
                    int ArgSize = Cuda->getNumArgOperands();

                    std::vector<Value *> Args1 ;
                    for (int X = 0; X < ArgSize; X++ ){
                    auto *Arg = Cuda->getArgOperand(X);
                    Value* Args = dyn_cast_or_null<Value>(Arg);
                    Args1.push_back(Args);
                    }
                    Instruction* NewInstruction = Builder.CreateCall(ConfigureFunction, Args1);
                    Value *Condition = Builder.CreateICmpNE(NewInstruction, One32Bit);
                    NewInstruction = SplitBlockAndInsertIfThen(
                    Condition, dyn_cast<Instruction>(Condition)->getNextNode(), false);
                    Builder.SetInsertPoint(NewInstruction);




                  
                    Type *TypeOfOutput = SizeOfTheOutput.second.first;
                    Function *Majority = M.getFunction("majorityVoting15");
                    if (Majority == nullptr)
                    Majority = createMajorityVoting(M, dyn_cast<PointerType>(TypeOfOutput));
                    std::vector<Value *> Args;
                    for(int i =0; i < CreatedOutputs.size(); i++)
                       errs() << *CreatedOutputs.at(i) << "+*+*\n";

                    errs() << *CreatedOutputs.at(CreatedOutputs.size() - 3)  << "\n";
                    errs() << *CreatedOutputs.at(CreatedOutputs.size() - 4)  << "\n";
                    errs() << *dyn_cast<CallInst>(NewFunctionCall)->getArgOperand(ParamSize-1) << "\n";

                    Args.push_back(
                      Builder.CreateLoad(
                        dyn_cast<Instruction>(dyn_cast<CallInst>(NewFunctionCall)->getArgOperand(ParamSize-1))->getOperand(0)
                        
                        )); // A
                    Args.push_back(Builder.CreateLoad(dyn_cast<Instruction>(CreatedOutputs.at(CreatedOutputs.size() - 4))->getOperand(0)));
                    Args.push_back(Builder.CreateLoad(dyn_cast<Instruction>(CreatedOutputs.at(CreatedOutputs.size() - 3))->getOperand(0)));    
                    Args.push_back(
                      Builder.CreateLoad(
                        dyn_cast<Instruction>(dyn_cast<CallInst>(NewFunctionCall)->getArgOperand(ParamSize-1))->getOperand(0)
                        
                        )); // A
                    Args.push_back( SizeOfTheOutput.first);

                   
                    Builder.CreateCall(Majority, Args);
                    
                  //Builder.CreateCall(F, Args);
                  
                CurrentInstruction++;
                CurrentInstruction++;
                
            }else if (FunctionName == "cudaConfigureCall") {
              Cuda = FunctionCall;
              LoadInst* BlockLoadInstr = dyn_cast_or_null<LoadInst>(Cuda->getArgOperand(0));
              if(BlockLoadInstr == nullptr)
                continue;
              IRBuilder<> Builder(BlockLoadInstr->getNextNode());
              Value* Mul = Builder.CreateMul(BlockLoadInstr, (ConstantInt::get(Int64Type, 3)));
              Cuda->setArgOperand(0, Mul);
              /*
              ICmpInst* CallConfig = dyn_cast_or_null<ICmpInst>(Cuda->getNextNode());
              errs() << *CallConfig->getOperand(1) << "///\n";
              CallConfig->setOperand(1, One32Bit);
*/
            }else if (FunctionName.contains("cudaMalloc")) {
              CudaMallocFunctionCalls.push_back(FunctionCall);  
          }
             }
      }
    }
    }

    return false;
    
  }

}; 

} // namespace
char RMTDevice::ID = -1;
char RMTHost::ID = -2;

static RegisterPass<RMTDevice> X("RMTDevice", "Hello World Pass", false, false);
static RegisterPass<RMTHost> XHost("RMTHost", "Hello World Pass", false, false);



static RegisterStandardPasses Y(PassManagerBuilder::EP_EarlyAsPossible,
                                [](const PassManagerBuilder &Builder,
                                   legacy::PassManagerBase &PM) {
                                  PM.add(new RMTDevice());
                                });

static RegisterStandardPasses YHost(PassManagerBuilder::EP_EarlyAsPossible,
                                [](const PassManagerBuilder &Builder,
                                   legacy::PassManagerBase &PM) {
                                  PM.add(new RMTHost());
                                });


