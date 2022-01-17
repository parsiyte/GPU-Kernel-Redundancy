#include "llvm/ADT/APInt.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Argument.h"
#include "llvm/IR/Attributes.h"
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
#include "llvm/Transforms/Utils/Local.h"
#include <algorithm>
#include <cstddef>
#include <string>
#include <utility>
#include <vector>
#include "llvm/Transforms/Utils/Cloning.h"
using namespace llvm;
#define NumberOfReplication 3
#define NumberofDimension 2
#define RevisitedSuffix "Revisited"

namespace {
struct RMTv2Device : public ModulePass {
  static char ID;
  RMTv2Device() : ModulePass(ID) {}


  StoreInst* getTheXID(Function* Kernel){
    StoreInst* ThreadIDX;
    CallInst* BlockIDX;
      for (Function::iterator BB = Kernel->begin(); BB != Kernel->end(); ++BB) {
        for (BasicBlock::iterator CurrentInstruction = BB->begin();CurrentInstruction != BB->end(); ++CurrentInstruction) {   
          if (CallInst *FunctionCall = dyn_cast<CallInst>(CurrentInstruction)) {
            StringRef FunctionName = FunctionCall->getCalledFunction()->getName();
            if(FunctionName == "llvm.nvvm.read.ptx.sreg.tid.x") 
              for (auto& U : FunctionCall->uses())
                 ThreadIDX = dyn_cast<StoreInst>(dyn_cast<Instruction> (U.getUser())->getNextNode());  
            else if(FunctionName == "llvm.nvvm.read.ptx.sreg.ctaid.x") {
              BlockIDX =  FunctionCall;
              }
        }
      }
      }
    return ThreadIDX;
  }

  void changeXID(Function *Kernel, Value* NewCreatedCall, Value *ToBeChange, IRBuilder<> Builder) {
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

    void changeYID(Function *Kernel, Value *ToBeChange) {
    for (Function::iterator BB = Kernel->begin(); BB != Kernel->end(); ++BB) {
      for (BasicBlock::iterator CurrentInstruction = BB->begin(); CurrentInstruction != BB->end(); ++CurrentInstruction) {
        if (CallInst *FunctionCall = dyn_cast<CallInst>(CurrentInstruction)) {
          StringRef FunctionName = FunctionCall->getCalledFunction()->getName();
          if (FunctionName == "llvm.nvvm.read.ptx.sreg.tid.y") {
            Instruction* Add = FunctionCall->getPrevNode();
            Instruction* BlockIdCall = Add->getPrevNode();
            BlockIdCall->replaceAllUsesWith(ToBeChange);
          } 
        }
      }
    }
  }  Function* createMajorityFuncton(Module &M, FunctionCallee MajorityVotingCallee){


    NamedMDNode *Annotations = M.getNamedMetadata("nvvm.annotations");
    LLVMContext &C = M.getContext();


    MDNode *N = MDNode::get(C, MDString::get(C, "kernel"));
    MDNode *TempN = MDNode::get(C, ConstantAsMetadata::get(ConstantInt::get(
                                       llvm::Type::getInt32Ty(C), 1)));
    // MDNode* TempA = MDNode::get(C,
    // ValueAsMetadata::get(MajorityVotingFunction));
    MDNode *Con = MDNode::concatenate(N, TempN);
    // Con = MDNode::concatenate(Con, TempA);

    Function *MajorityVotingFunction =
        dyn_cast<Function>(MajorityVotingCallee.getCallee());
    MajorityVotingFunction->setCallingConv(CallingConv::C);



  FunctionCallee GetThread = M.getFunction("llvm.nvvm.read.ptx.sreg.tid.x");

FunctionCallee GetBlockDim =
        M.getOrInsertFunction("llvm.nvvm.read.ptx.sreg.ntid.x", GetThread.getFunctionType());
    FunctionCallee GetGridDim =
        M.getFunction("llvm.nvvm.read.ptx.sreg.ctaid.x");

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


    Annotations->addOperand(MDNode::concatenate(
        MDNode::get(C, ValueAsMetadata::get(MajorityVotingFunction)), Con));

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
      Function* BlockIDX = M.getFunction("llvm.nvvm.read.ptx.sreg.ctaid.x");
      Function* BlockIDY = M.getFunction("llvm.nvvm.read.ptx.sreg.ctaid.y");

     for(unsigned int Index = 0; Index < Annotations->getNumOperands(); Index++){
        MDNode* Operand = Annotations->getOperand(Index);
        Metadata* Feature = Operand->getOperand(1).get();
        if(cast<MDString>(Feature)->getString()  == "kernel"){        
          
        Metadata* FunctionMetadata = cast<Metadata>(Operand->getOperand(0));
        ValueAsMetadata* AsValue = cast<ValueAsMetadata>(FunctionMetadata);
        Function* Kernel =  dyn_cast<Function>(AsValue->getValue());
        unsigned int ArgumanCount = Kernel->arg_size();
        Value* PossibleOutput = Kernel->getArg(ArgumanCount-1);
        Type* PossibleOutputType = PossibleOutput->getType();
        
        if(Kernel->getName().contains("Revisited") || Kernel->getName().contains("majorityVoting15") || !PossibleOutputType->isPointerTy())
          continue;
          
        AttributeList Attributes = Kernel->getAttributes();
        errs() << Kernel->getFnAttribute("correctly-rounded-divide-sqrt-fp-math").getAsString() << "*\n";


        FunctionType* KernelType = Kernel->getFunctionType();

        std::vector<Type *> NewFunctionType;
        for(unsigned int SIndex = 0; SIndex < ArgumanCount; SIndex++ ){
          NewFunctionType.push_back(KernelType->getParamType(SIndex));
        }
        for(int OutputIndex = 0; OutputIndex < NumberOfReplication - 1; OutputIndex++){
          NewFunctionType.push_back(PossibleOutputType);
        }

        NewFunctionType.push_back(Int32Type);
        FunctionType* NewKernelType = FunctionType::get(VoidType,NewFunctionType, true);
        std::string NewKernelName = Kernel->getName().str() + "Revisited";
        
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


        Value* Output = NewKernelFunction->getArg(ArgumanCount - 1);
        User* FirstUser =  Output->uses().begin()->getUser();
        StoreInst* OutputStore = dyn_cast<StoreInst>(FirstUser);
        AllocaInst* OutputAllocation = dyn_cast<AllocaInst>(OutputStore->getPointerOperand());

       BasicBlock* FirstBB = dyn_cast<BasicBlock>(NewKernelFunction->begin());
       
       Instruction& LastInstruction = FirstBB->front();

        Function::arg_iterator Args = NewKernelFunction->arg_end();
        Args--;
        //Value *OriginalY = Args--;
        Value *OriginalBased = Args--;
        Value *SecondRedundantArg = Args--;   
        Value *FirstRedundantArg = Args--;
        Value *OriginalOutput = Args--;
        std::string ValueName = OriginalOutput->getName().str();
        FirstRedundantArg->setName(ValueName + "1");
        SecondRedundantArg->setName(ValueName + "2");
        OriginalBased->setName("OriginalBased");

        IRBuilder<> Builder(OutputAllocation->getNextNode());

        Instruction* FirstRedundant =  Builder.CreateAlloca(PossibleOutputType,nullptr, "FirstRedundant");  
        Instruction* SecondRedundant =  Builder.CreateAlloca(PossibleOutputType,nullptr, "SecondRedundant");
        Value* OriginalBaseddr = Builder.CreateAlloca(Int32Type,nullptr, "OriginalBased.addr");
        //Value* OriginalYAddr = Builder.CreateAlloca(Int32Type,nullptr, "OriginalY.addr");

        Builder.CreateStore(OriginalBased, OriginalBaseddr);
        Builder.CreateStore(FirstRedundantArg, FirstRedundant);
        Builder.CreateStore(SecondRedundantArg, SecondRedundant);
        

        //errs() << *OutputAllocation << "\n";
        //IRBuilder<> Builder(OutputAllocation->getNextNode());
        Instruction* MetaOutput = Builder.CreateAlloca(PossibleOutputType,nullptr, "MetaOutput");  
        OutputAllocation->replaceAllUsesWith(MetaOutput);
        

        Builder.CreateStore(Output,OutputAllocation );
        

        Instruction* BlockIdaddr =  Builder.CreateAlloca(Int32Type,nullptr, "BlockIdaddr");
        Instruction* BlockIdaddr2 =  Builder.CreateAlloca(Int32Type,nullptr, "BlockIdaddr2");
        Instruction* BlockIdaddrY =  Builder.CreateAlloca(Int32Type,nullptr, "BlockIdaddrY");
        

        
  
        Value* BlockIDYCall;
         Value* BlockYID2 ;
        Value* BlockIDXCall = Builder.CreateCall(BlockIDX);
        /*
        if(BlockIDY != nullptr){
        BlockIDYCall = Builder.CreateCall(BlockIDY);
        BlockYID2 = Builder.CreateURem(
            BlockIDYCall,
            Builder.CreateLoad(OriginalYAddr)
         );
        Builder.CreateStore(BlockYID2, BlockIdaddrY);

         changeYID(NewKernelFunction, Builder.CreateLoad(BlockIdaddrY));
        }*/
    
        Value* BlockID = Builder.CreateUDiv(
            BlockIDXCall,
            Builder.CreateLoad(OriginalBaseddr)
         );

        BlockID->setName("BlockID");
        Builder.CreateStore(BlockID, BlockIdaddr);

        Value* BlockID2 = Builder.CreateURem(
            BlockIDXCall,
            Builder.CreateLoad(OriginalBaseddr)
         );
         BlockID2->setName("BlockID2");
        Builder.CreateStore(BlockID2, BlockIdaddr2);
      
         StoreInst*  I =  getTheXID(NewKernelFunction);

      
        errs() << *I << "\n";

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
    AllocaInst *PrintFAlloca = Builder.CreateAlloca(d->getScalarType());
     auto *XX =  Builder.CreateInBoundsGEP(PrintFAlloca, {ConstantInt::get(Type::getInt32Ty(Context),0),ConstantInt::get(Type::getInt32Ty(Context),0)});
     Builder.CreateStore(BlockID, XX);
        Builder.CreateCall(
            M.getFunction("vprintf"),
            {Builder.CreateGlobalStringPtr("%d \n"),
             Builder.CreateBitCast(PrintFAlloca, Int8ptrType)
             });


*/
         changeXID(NewKernelFunction,BlockIDXCall,BlockIdaddr2, Builder);


         
        Builder.SetInsertPoint(OutputStore->getNextNode());
        BlockID = Builder.CreateLoad(BlockIdaddr);
        Instruction* ZeroCmp = dyn_cast<Instruction>(Builder.CreateICmpEQ(BlockID, Zero32Bit));
        Instruction *ThenTerm, *FirstElseIfCondTerm;
        SplitBlockAndInsertIfThenElse(ZeroCmp, ZeroCmp->getNextNode(), &ThenTerm, &FirstElseIfCondTerm); 
        Builder.SetInsertPoint(ThenTerm);
        Builder.CreateStore(Builder.CreateLoad(OutputAllocation), MetaOutput);


        Instruction *ElseIfTerm, *SecondElseTerm;
        Builder.SetInsertPoint(FirstElseIfCondTerm);
        BlockID = Builder.CreateLoad(BlockIdaddr);
        Instruction* OneCmp = dyn_cast<Instruction>(Builder.CreateICmpEQ(BlockID, One32Bit));
        SplitBlockAndInsertIfThenElse(OneCmp, OneCmp->getNextNode(), &ElseIfTerm, &SecondElseTerm); 
        Builder.SetInsertPoint(ElseIfTerm);
        Builder.CreateStore(Builder.CreateLoad(FirstRedundant), MetaOutput);


        Builder.SetInsertPoint(SecondElseTerm);
        BlockID = Builder.CreateLoad(BlockIdaddr);
        Instruction* TwoCmp = dyn_cast<Instruction>(Builder.CreateICmpEQ(BlockID, Two32Bit));
        Instruction* NewBranch  = SplitBlockAndInsertIfThen(TwoCmp, TwoCmp->getNextNode(), false);
        Builder.SetInsertPoint(NewBranch);
        Builder.CreateStore(Builder.CreateLoad(SecondRedundant), MetaOutput);


        MDNode *N = MDNode::get(Context, MDString::get(Context, "kernel"));
        MDNode *TempN = MDNode::get(Context, ConstantAsMetadata::get(ConstantInt::get(Int32Type, 1)));
        MDNode *Con = MDNode::concatenate(N, TempN);
        Annotations->addOperand(MDNode::concatenate(MDNode::get(Context, ValueAsMetadata::get(NewKernelFunction)), Con));

      FunctionCallee MajorityVotingCallee = M.getOrInsertFunction(
          "majorityVoting2", Type::getVoidTy(Context),
          Type::getFloatPtrTy(Context), Type::getFloatPtrTy(Context),
          Type::getFloatPtrTy(Context),
          Type::getInt64Ty(Context)

      );
      createMajorityFuncton(M,MajorityVotingCallee );


        //Builder.CreateStore(Builder.CreateLoad(FirstRedundant), TempRedundant);

     }
     }

    return false;
  }

};

struct RMTv2Host : public ModulePass {
  static char ID;

  struct CudaConfigurations{
    Value* OriginalX;
    Value* OriginalY;
    CallInst* ConfigurationCall;

  };
  RMTv2Host() : ModulePass(ID) {}

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
                  std::string Output, IRBuilder<> *Builder) {
    Value *Size = nullptr;
    std::pair<Type *, Type *> Types;
    for (size_t Index = 0; Index < CudaMallocFunctionCalls.size(); Index++) {
      CallInst *CudaMallocFunctionCall = CudaMallocFunctionCalls.at(Index);
      AllocaInst *AllocaVariable = nullptr;
      Type *DestinationType = nullptr;
      if (BitCastInst *BitCastVariable = dyn_cast<BitCastInst>(CudaMallocFunctionCall->getArgOperand(0))) {
        AllocaVariable = dyn_cast<AllocaInst>(BitCastVariable->getOperand(0));
        DestinationType = dyn_cast<PointerType>(BitCastVariable->getDestTy())->getElementType();
      } else if (dyn_cast_or_null<AllocaInst>(CudaMallocFunctionCall->getArgOperand(0)) != nullptr) {
        AllocaVariable = dyn_cast<AllocaInst>(CudaMallocFunctionCall->getArgOperand(0));
        DestinationType = dyn_cast<PointerType>(AllocaVariable->getAllocatedType());
      }
      std::string VariableName = AllocaVariable->getName().str();
      if (VariableName == Output) {
        Builder->SetInsertPoint(CudaMallocFunctionCall);
        Size = CudaMallocFunctionCall->getArgOperand(1);
        Types =std::make_pair(AllocaVariable->getAllocatedType(), DestinationType);
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


Value* findThreadX(BasicBlock* BB){

      Function* MainFunction = BB->getParent();
      Type* Int32Type = Type::getInt32Ty(BB->getContext());
        for (BasicBlock::iterator CurrentInstruction = BB->begin(); CurrentInstruction != BB->end(); ++CurrentInstruction) {
          if(CallInst* FunctionCall = dyn_cast<CallInst>(CurrentInstruction) ){
               StringRef FunctionName = FunctionCall->getCalledFunction()->getName();
               if(FunctionName == "cudaConfigureCall"){
                LoadInst* GridLoadInstr = dyn_cast_or_null<LoadInst>(FunctionCall->getArgOperand(2)); // 0 Grid X oluyor

                GetElementPtrInst* GEPS =  dyn_cast_or_null<GetElementPtrInst>(GridLoadInstr->getPointerOperand()); // Grid {i64, i32} şekilde bir struct.
                AllocaInst* GridAlloca = dyn_cast_or_null<AllocaInst>(GEPS->getPointerOperand()); // Grid'in alloca instruction'ı
                for (Use& U : GridAlloca->uses()) {
                  User* User = U.getUser();
                  BitCastInst* Casted = dyn_cast_or_null<BitCastInst>(User);
                  if(Casted != nullptr){
                    Instruction* TmpInstruction = Casted;
                    MemCpyInst* GridMemoryCopy = nullptr;
                    while(GridMemoryCopy == nullptr){
                      TmpInstruction = TmpInstruction->getNextNode();
                      GridMemoryCopy = dyn_cast_or_null<MemCpyInst>(TmpInstruction);
                    }
                    BitCastInst* CopySrc = dyn_cast_or_null<BitCastInst>(GridMemoryCopy->getArgOperand(1));
                    AllocaInst* GridTotalAllocation  = dyn_cast_or_null<AllocaInst>(CopySrc->getOperand(0));
                    errs() << *GridTotalAllocation << "\n";
                    bool Found = false;
                    for(Function::iterator BasicB = MainFunction->begin();  BasicB != MainFunction->end(); BasicB++)
                      for (BasicBlock::iterator CurrentInstruction2 = BasicB->begin(); CurrentInstruction2 != BasicB->end(); ++CurrentInstruction2) {
                          if( MemCpyInst* TempMemCpy =  dyn_cast<MemCpyInst>(CurrentInstruction2 )){
                            BitCastInst* PossibleGridX = dyn_cast_or_null<BitCastInst>(TempMemCpy->getArgOperand(0));
                            AllocaInst* PossibleGridAllocation  = dyn_cast_or_null<AllocaInst>(PossibleGridX->getOperand(0));
                            if(PossibleGridAllocation  == GridTotalAllocation){
                                  BitCastInst* CopyDestination = dyn_cast_or_null<BitCastInst>(TempMemCpy->getArgOperand(1));
                                  AllocaInst* DestinationAllocation  = dyn_cast_or_null<AllocaInst>(CopyDestination->getOperand(0));
                              for(Function::iterator BasicB1 = MainFunction->begin();  BasicB1 != MainFunction->end(); BasicB1++)
                                  for (BasicBlock::iterator CurrentInstruction3 = BasicB1->begin(); CurrentInstruction3 != BasicB1->end(); ++CurrentInstruction3) {
                                    if( CallInst* FuncCall =  dyn_cast<CallInst>(CurrentInstruction3 )){
                                        StringRef FunctionName = FuncCall->getCalledFunction()->getName();
                                        if(FunctionName.contains("dim3") == true) {
                                      errs() << *FuncCall << "\n";
                                        if( FuncCall->getArgOperand(0) ==  DestinationAllocation){
                                          Value* GridX = FuncCall->getArgOperand(1);
                                           return GridX;
                                        }
                                      }}
                                  }

                            }


                          }

                      }

                  }
                }
               }
          }
}
}
Value* findThreadY(BasicBlock* BB){

      Function* MainFunction = BB->getParent();
      Type* Int32Type = Type::getInt32Ty(BB->getContext());
        for (BasicBlock::iterator CurrentInstruction = BB->begin(); CurrentInstruction != BB->end(); ++CurrentInstruction) {
          if(CallInst* FunctionCall = dyn_cast<CallInst>(CurrentInstruction) ){
               StringRef FunctionName = FunctionCall->getCalledFunction()->getName();
               if(FunctionName == "cudaConfigureCall"){
                LoadInst* GridLoadInstr = dyn_cast_or_null<LoadInst>(FunctionCall->getArgOperand(3)); // 0 Grid X oluyor

                GetElementPtrInst* GEPS =  dyn_cast_or_null<GetElementPtrInst>(GridLoadInstr->getPointerOperand()); // Grid {i64, i32} şekilde bir struct.
                AllocaInst* GridAlloca = dyn_cast_or_null<AllocaInst>(GEPS->getPointerOperand()); // Grid'in alloca instruction'ı
                for (Use& U : GridAlloca->uses()) {
                  User* User = U.getUser();
                  BitCastInst* Casted = dyn_cast_or_null<BitCastInst>(User);
                  if(Casted != nullptr){
                    Instruction* TmpInstruction = Casted;
                    MemCpyInst* GridMemoryCopy = nullptr;
                    while(GridMemoryCopy == nullptr){
                      TmpInstruction = TmpInstruction->getNextNode();
                      GridMemoryCopy = dyn_cast_or_null<MemCpyInst>(TmpInstruction);
                    }
                    BitCastInst* CopySrc = dyn_cast_or_null<BitCastInst>(GridMemoryCopy->getArgOperand(1));
                    AllocaInst* GridTotalAllocation  = dyn_cast_or_null<AllocaInst>(CopySrc->getOperand(0));
                    errs() << *GridTotalAllocation << "\n";
                    bool Found = false;
                    for(Function::iterator BasicB = MainFunction->begin();  BasicB != MainFunction->end(); BasicB++)
                      for (BasicBlock::iterator CurrentInstruction2 = BasicB->begin(); CurrentInstruction2 != BasicB->end(); ++CurrentInstruction2) {
                          if( MemCpyInst* TempMemCpy =  dyn_cast<MemCpyInst>(CurrentInstruction2 )){
                            BitCastInst* PossibleGridX = dyn_cast_or_null<BitCastInst>(TempMemCpy->getArgOperand(0));
                            AllocaInst* PossibleGridAllocation  = dyn_cast_or_null<AllocaInst>(PossibleGridX->getOperand(0));
                            if(PossibleGridAllocation  == GridTotalAllocation){
                                  BitCastInst* CopyDestination = dyn_cast_or_null<BitCastInst>(TempMemCpy->getArgOperand(1));
                                  AllocaInst* DestinationAllocation  = dyn_cast_or_null<AllocaInst>(CopyDestination->getOperand(0));
                              for(Function::iterator BasicB1 = MainFunction->begin();  BasicB1 != MainFunction->end(); BasicB1++)
                                  for (BasicBlock::iterator CurrentInstruction3 = BasicB1->begin(); CurrentInstruction3 != BasicB1->end(); ++CurrentInstruction3) {
                                    if( CallInst* FuncCall =  dyn_cast<CallInst>(CurrentInstruction3 )){
                                        StringRef FunctionName = FuncCall->getCalledFunction()->getName();
                                        if(FunctionName.contains("dim3") == true) {
                                      errs() << *FuncCall << "\n";
                                        if( FuncCall->getArgOperand(0) ==  DestinationAllocation){
                                          Value* GridX = FuncCall->getArgOperand(2);
                                          errs() << *GridX << "YY\n";
                                           return GridX;
                                        }
                                      }}
                                  }

                            }


                          }

                      }

                  }
                }
               }
          }
 }
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


struct CudaConfigurations findConfigurationCall(BasicBlock* BB){
      struct CudaConfigurations Configurations;
      Function* MainFunction = BB->getParent();
      Type* Int32Type = Type::getInt32Ty(BB->getContext());
        for (BasicBlock::iterator CurrentInstruction = BB->begin(); CurrentInstruction != BB->end(); ++CurrentInstruction) {
          if(CallInst* FunctionCall = dyn_cast<CallInst>(CurrentInstruction) ){
               StringRef FunctionName = FunctionCall->getCalledFunction()->getName();
               if(FunctionName == "cudaConfigureCall"){
                LoadInst* GridLoadInstr = dyn_cast_or_null<LoadInst>(FunctionCall->getArgOperand(0)); // 0 Grid X oluyor

               Configurations.ConfigurationCall = FunctionCall;
                GetElementPtrInst* GEPS =  dyn_cast_or_null<GetElementPtrInst>(GridLoadInstr->getPointerOperand()); // Grid {i64, i32} şekilde bir struct.
                AllocaInst* GridAlloca = dyn_cast_or_null<AllocaInst>(GEPS->getPointerOperand()); // Grid'in alloca instruction'ı
                for (Use& U : GridAlloca->uses()) {
                  User* User = U.getUser();
                  BitCastInst* Casted = dyn_cast_or_null<BitCastInst>(User);
                  if(Casted != nullptr){
                    Instruction* TmpInstruction = Casted;
                    MemCpyInst* GridMemoryCopy = nullptr;
                    while(GridMemoryCopy == nullptr){
                      TmpInstruction = TmpInstruction->getNextNode();
                      GridMemoryCopy = dyn_cast_or_null<MemCpyInst>(TmpInstruction);
                    }
                    BitCastInst* CopySrc = dyn_cast_or_null<BitCastInst>(GridMemoryCopy->getArgOperand(1));
                    AllocaInst* GridTotalAllocation  = dyn_cast_or_null<AllocaInst>(CopySrc->getOperand(0));

                    bool Found = false;
                    for(Function::iterator BasicB = MainFunction->begin();  BasicB != MainFunction->end(); BasicB++)
                      for (BasicBlock::iterator CurrentInstruction2 = BasicB->begin(); CurrentInstruction2 != BasicB->end(); ++CurrentInstruction2) {
                          if( MemCpyInst* TempMemCpy =  dyn_cast<MemCpyInst>(CurrentInstruction2 )){
                            BitCastInst* PossibleGridX = dyn_cast_or_null<BitCastInst>(TempMemCpy->getArgOperand(0));
                            AllocaInst* PossibleGridAllocation  = dyn_cast_or_null<AllocaInst>(PossibleGridX->getOperand(0));
                            if(PossibleGridAllocation  == GridTotalAllocation){
                                  BitCastInst* CopyDestination = dyn_cast_or_null<BitCastInst>(TempMemCpy->getArgOperand(1));
                                  AllocaInst* DestinationAllocation  = dyn_cast_or_null<AllocaInst>(CopyDestination->getOperand(0));
                              for(Function::iterator BasicB1 = MainFunction->begin();  BasicB1 != MainFunction->end(); BasicB1++)
                                  for (BasicBlock::iterator CurrentInstruction3 = BasicB1->begin(); CurrentInstruction3 != BasicB1->end(); ++CurrentInstruction3) {
                                    if( CallInst* FuncCall =  dyn_cast<CallInst>(CurrentInstruction3 )){
                                        StringRef FunctionName = FuncCall->getCalledFunction()->getName();
                                        if(FunctionName.contains("dim3") == true && FuncCall->hasMetadata("Multiplication") == false)  {
                                          if( FuncCall->getArgOperand(0) ==  DestinationAllocation){
                                            Value* GridX = FuncCall->getArgOperand(1);
                                            Value* GridY = FuncCall->getArgOperand(2);
                                            Instruction* GridXAsInstruction = dyn_cast_or_null<Instruction>(GridX);
                                            Instruction* GridYAsInstruction = dyn_cast_or_null<Instruction>(GridY);

                                            IRBuilder<> Builder(FuncCall->getPrevNode());
                                            if(GridXAsInstruction != nullptr)
                                                Builder.SetInsertPoint(GridXAsInstruction->getNextNode());

                                            Constant *Three = ConstantInt::get(Int32Type, 3);
                                            Configurations.OriginalX = GridX;
                                            FuncCall->setArgOperand(1, Builder.CreateMul(GridX, Three));
                                            if(GridYAsInstruction != nullptr)
                                                Builder.SetInsertPoint(GridYAsInstruction->getNextNode());
                                            Configurations.OriginalY = GridY;

    
                                            LLVMContext& C = FuncCall->getContext();
                                            MDNode* N = MDNode::get(C, MDString::get(C, "true"));
                                            FuncCall->setMetadata("Multiplication", N);
                                            break;
                                          }
                                        }
                                      }
                                  }

                            }


                          }

                      }

                  }
                }
               }
          }
       } 

  return Configurations;
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

void createAndAllocateVariableAndreMemCpy(IRBuilder<> Builder, Output* OutputToReplicate, Instruction& LastInstructionOfPrevBB, FunctionCallee CudaMemcpy, bool IsLoop){
  LLVMContext &Context = LastInstructionOfPrevBB.getContext();
  Type *Int32Type = Type::getInt32Ty(Context);
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
    Value* BitcastedCloned = Builder.CreateBitCast(NewAllocated, DestinationType);
    Cloned->setArgOperand(0, BitcastedCloned);
    Cloned->insertAfter(OutputToReplicate->MallocInstruction);
    Builder.SetInsertPoint(&LastInstructionOfPrevBB); 
    BitcastedCloned = Builder.CreateBitCast(Builder.CreateLoad(NewAllocated), DestinationType->getPointerElementType());
    Value* LoadedOutput = Builder.CreateLoad(OutputToReplicateAllocation);
    Value* BitcastedOutput = Builder.CreateBitCast(LoadedOutput, DestinationType->getPointerElementType());
    Builder.CreateCall(CudaMemcpy, {BitcastedCloned, BitcastedOutput, Size, Three32Bit});
    OutputToReplicate->Replications.push_back(NewAllocated);
  }

}

  Function *createRevisted(Module &M,std::string FunctionName,FunctionCallee RevistedFunctionCallee) {
    // std::to_string(RevistedPointerType->getTypeID());
    LLVMContext &Context = M.getContext();
    Type *Int64Type = Type::getInt64Ty(Context);
    Type *Int32Type = Type::getInt32Ty(Context);
    Type *FloatType = Type::getFloatTy(Context);
    Value *Zero32Bit = ConstantInt::get(Int32Type, 0);
    PointerType *Int8PtrType = Type::getInt8PtrTy(Context);
    PointerType *Int32PtrType = Type::getInt32PtrTy(Context);

    Function *CudaRegisterFunction = M.getFunction("__cuda_register_globals");
    Function *CudaRegisterFunction2 = M.getFunction("__cudaRegisterFunction");
    Function *CudaSetupArgument = M.getFunction("cudaSetupArgument");
    ConstantPointerNull* Null8bitPtr = ConstantPointerNull::get(Int8PtrType);
    ConstantPointerNull* Null32bitPtr = ConstantPointerNull::get(Int32PtrType);

    Function *CudaLaunch = M.getFunction("cudaLaunch");


    std::vector<Value *> Parameters;

    Function *RevistedFunction =  dyn_cast<Function>(RevistedFunctionCallee.getCallee());
    RevistedFunction->setCallingConv(CallingConv::C);
    Function::arg_iterator Args = RevistedFunction->arg_begin();

     BasicBlock *EntryBlock =
        BasicBlock::Create(Context, "entry", RevistedFunction);

    IRBuilder<> Builder(EntryBlock);

    Builder.SetInsertPoint(EntryBlock);
    int Offset = 0;
    int SizeParameter = 8;
    while(Args != RevistedFunction->arg_end()){
      Value* Arguman = Args++;
      Type* ArgumanType = Arguman->getType();
      if (ArgumanType == Int32PtrType || ArgumanType == Int32Type || ArgumanType == FloatType)
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

       Value *SizeValue = ConstantInt::get(Int64Type, SizeParameter);
      Value *CudaSetupArgumentCall = Builder.CreateCall(
          CudaSetupArgument, {BitcastParameter, SizeValue, OffsetValue});
      Instruction *IsError = dyn_cast<Instruction>(
          Builder.CreateICmpEQ(CudaSetupArgumentCall, Zero32Bit));
      if (Offset == 0)
        Builder.CreateRetVoid(); // Buraya daha akıllıca çözüm bulmak gerekiyor.
                                 // Sevimli gözükmüyor.

      Instruction *SplitPoint =  SplitBlockAndInsertIfThen(IsError, IsError->getNextNode(), false);

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
         Null8bitPtr ,Null8bitPtr,Null8bitPtr,Null8bitPtr, Null32bitPtr});

    return RevistedFunction;
  }



FunctionType * getTheNewKernelType(FunctionType* OriginalFunctionType, std::vector<Output> Outputs){

  FunctionType* NewKernelFunctionType;
  size_t NumberOfArg =  OriginalFunctionType->getNumParams();
  LLVMContext & Context = OriginalFunctionType->getContext();
  Type* Int32Type = Type::getInt32Ty(Context);

  std::vector<Type * > NewKernelTypes;
  for(size_t ArgIndex = 0; ArgIndex < NumberOfArg; ArgIndex++){
    NewKernelTypes.push_back(OriginalFunctionType->getFunctionParamType(ArgIndex));
  }

  for( Output SingleOutput : Outputs){
     Type* SingleOutputType = SingleOutput.OutputType;
     for(int TempIndex = 0; TempIndex < NumberOfReplication - 1; TempIndex++){
       NewKernelTypes.push_back(SingleOutputType);
     }
  }

  NewKernelTypes.push_back(Int32Type);
    
  NewKernelFunctionType = FunctionType::get(OriginalFunctionType->getReturnType(), NewKernelTypes, true); 
  
  return NewKernelFunctionType;
}



  bool runOnModule(Module &M) override {
    CallInst *ConfigurationCall;

    struct CudaConfigurations Configurations;
    Function *CudaMalloc = M.getFunction("cudaMalloc");
    Function *CudaFree = M.getFunction("cudaFree");
    Function *CudaMemCpy = M.getFunction("cudaMemcpy");
    LLVMContext &Context = M.getContext();
    Type *Int8PtrType = Type::getInt8PtrTy(Context);
    Type *Int32Type = Type::getInt32Ty(Context);
    Type *Int64Type = Type::getInt64Ty(Context);
    Type *VoidType = Type::getVoidTy(Context);
    std::vector<CallInst *> CudaMallocFunctionCalls;
    std::vector<CallInst *> CudaMemcpyFunctionCalls;
    Value *One32Bit = ConstantInt::get(Int32Type, 1);
    Value *Zero32Bit = ConstantInt::get(Int32Type, 0);

    Function *ConfigureFunction = M.getFunction("cudaConfigureCall");

    Function *Sync = M.getFunction("cudaThreadSynchronize");
    std::vector<Value *> CreatedOutputs;
    for (Module::iterator F = M.begin(); F != M.end(); ++F) {
      for (Function::iterator BB = F->begin(); BB != F->end(); ++BB) {
        for (BasicBlock::iterator CurrentInstruction = BB->begin(); CurrentInstruction != BB->end(); ++CurrentInstruction) {
          if (CallInst *FunctionCall = dyn_cast<CallInst>(CurrentInstruction)) {
            StringRef FunctionName = FunctionCall->getCalledFunction()->getName();
            if (isReplicate(FunctionCall)) {


              BasicBlock* CurrentBB = dyn_cast<BasicBlock>(BB);
              BasicBlock* NextBB = CurrentBB->getNextNode();
              BasicBlock* PrevBB = CurrentBB->getPrevNode();


              Instruction* FirstInstructionOfNextBB = NextBB->getFirstNonPHI();
              Instruction* FirstInstructionOfPrevBB = PrevBB->getFirstNonPHI();
              Instruction& LastInstructionOfPrevBB = PrevBB->back();

              Function* FunctionToReplicate = FunctionCall->getCalledFunction();

              bool IsLoop = isForLoop(PrevBB);

              Configurations = findConfigurationCall(PrevBB);
              ConfigurationCall = Configurations.ConfigurationCall;

              IRBuilder<> Builder(FunctionCall);  

              MDNode *RedundancyMetadata = FunctionCall->getMetadata("Redundancy");
              StringRef MetadataString =  getMetadataString(RedundancyMetadata);
              std::pair<std::vector<std::string>, std::vector<std::string>> InputsAndOutputs = parseMetadataString(MetadataString);
              std::vector<std::string> Inputs = InputsAndOutputs.first;
              std::vector<std::string> Outputs = InputsAndOutputs.second;
              size_t NumberOfArg = FunctionCall->arg_size();
              std::vector<Value *> NewArgs;
              std::vector<Output > OutputsToBeReplicated;

              for(size_t ArgIndex = 0; ArgIndex < NumberOfArg; ArgIndex++){
                NewArgs.push_back(FunctionCall->getArgOperand(ArgIndex));
              }


              for (size_t Index = 0; Index < Outputs.size(); Index++) {
                  std::string VariableName = Outputs[Index];
                  Output SingleOutput;
                  SingleOutput.Name = VariableName;                  
                  parseOutput(CudaMallocFunctionCalls, &SingleOutput);
                  createAndAllocateVariableAndreMemCpy(Builder, &SingleOutput, LastInstructionOfPrevBB, CudaMemCpy, IsLoop);
                  OutputsToBeReplicated.push_back(SingleOutput);
              }


              Builder.SetInsertPoint(FunctionCall);
              for(Output &CreatedOutput : OutputsToBeReplicated){
                  std::vector<Instruction *> Replications =  CreatedOutput.Replications;
                  for(Instruction* Replication : Replications){
                    Value* LoadedNewArg = Builder.CreateLoad(Replication);
                    NewArgs.push_back(LoadedNewArg);
                 }
              }

              NewArgs.push_back(Configurations.OriginalX);
              //NewArgs.push_back(Configurations.OriginalY);

              FunctionType* NewKernelType = getTheNewKernelType(FunctionCall->getFunctionType(), OutputsToBeReplicated);
              std::string NewKernelName = FunctionName.str() + RevisitedSuffix;
              
              FunctionCallee NewKernelAsCallee =  M.getOrInsertFunction(NewKernelName, NewKernelType);
              Instruction* NewFunctionCall = Builder.CreateCall(NewKernelAsCallee, NewArgs);

              createRevisted(M, NewKernelName, NewKernelAsCallee);

              CurrentInstruction++;
              FunctionCall->eraseFromParent();  

                CurrentBB = NewFunctionCall->getParent();
                NextBB = CurrentBB->getNextNode();
                PrevBB = CurrentBB->getPrevNode();
                Instruction *FirstInstruction = dyn_cast<Instruction>(PrevBB->begin());
                FirstInstruction = dyn_cast<Instruction>(NextBB->begin());

                              
              for(size_t OutputIndex = 0; OutputIndex < OutputsToBeReplicated.size(); OutputIndex++ ){

                Instruction* ClonedConfigure = ConfigurationCall->clone();
                ClonedConfigure->insertAfter(FirstInstruction);
                Builder.SetInsertPoint(ClonedConfigure->getNextNode());
                Instruction* ConfigureCheck = dyn_cast<Instruction>(Builder.CreateICmpNE(ClonedConfigure, One32Bit));
                
                Instruction* FirstInstructionOfNextBB = SplitBlockAndInsertIfThen(ConfigureCheck, ConfigureCheck->getNextNode(), false);
                Builder.SetInsertPoint(FirstInstructionOfNextBB);


                Output CurrentOutput = OutputsToBeReplicated.at(OutputIndex);
                Type* OutputType = CurrentOutput.OutputType;
                std::string MajorityFunctionName = "majorityFunction" + std::to_string(OutputType->getTypeID());
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

/*
              createRevisted(Module &M, std::string FunctionName, FunctionCallee RevistedFunctionCallee)

             for(int ReplicationIndex = 1; ReplicationIndex < NumberOfReplication; ReplicationIndex++){
                CallInst* ClonedConfigureCall = dyn_cast<CallInst>(ConfigurationCall->clone());
                ClonedConfigureCall->insertBefore(FirstInstructionOfNextBB);  

                  Builder.SetInsertPoint(ClonedConfigureCall->getNextNode());
                  Instruction* ConfgurationCheck = dyn_cast<Instruction>(Builder.CreateICmpNE(ClonedConfigureCall, One32Bit));
                  Instruction* NewBasicBlockFirstInstruction = SplitBlockAndInsertIfThen(ConfgurationCheck, ConfgurationCheck->getNextNode(), false);
                  errs() << *NewBasicBlockFirstInstruction << "\n";

                  int NumberOfArg = FunctionCall->getNumArgOperands();
                  std::vector<Value *> ArgsOfReplicationFunction;
                  for(int ArgIndex = 0; ArgIndex < NumberOfArg - 1; ArgIndex++){ // Outputu çıkartıyoruz
                    Value* Arg = FunctionCall->getArgOperand(ArgIndex);
                    if(Instruction* ArgAsInstruction = dyn_cast<Instruction>(Arg)){
                      Instruction* ClonedArg = ArgAsInstruction->clone();
                      ClonedArg->insertBefore(NewBasicBlockFirstInstruction);
                      Arg = ClonedArg;
                    }
                    ArgsOfReplicationFunction.push_back(Arg);
                  }
                  Builder.SetInsertPoint(NewBasicBlockFirstInstruction);
                  
                  for(size_t OutputIndex = 0; OutputIndex < OutputsToBeReplicated.size(); OutputIndex++){
                    Instruction* NewOutput = OutputsToBeReplicated.at(OutputIndex).Replications.at(ReplicationIndex-1);
                    ArgsOfReplicationFunction.push_back(Builder.CreateLoad(NewOutput));
                  }

                  Builder.CreateCall(FunctionToReplicate, ArgsOfReplicationFunction);
                  }

*/














            } else if (FunctionName == "cudaMemcpy") {
              CudaMemcpyFunctionCalls.push_back(FunctionCall);

            } else if (FunctionName.contains("cudaMalloc")) {
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
char RMTv2Device::ID = -1;
char RMTv2Host::ID = -2;

static RegisterPass<RMTv2Device> X("RMTv2Device", "Hello World Pass", false, false);
static RegisterPass<RMTv2Host> XHost("RMTv2Host", "Hello World Pass", false, false);



static RegisterStandardPasses Y(PassManagerBuilder::EP_EarlyAsPossible,
                                [](const PassManagerBuilder &Builder,
                                   legacy::PassManagerBase &PM) {
                                  PM.add(new RMTv2Device());
                                });

static RegisterStandardPasses YHost(PassManagerBuilder::EP_EarlyAsPossible,
                                [](const PassManagerBuilder &Builder,
                                   legacy::PassManagerBase &PM) {
                                  PM.add(new RMTv2Host());
                                });

