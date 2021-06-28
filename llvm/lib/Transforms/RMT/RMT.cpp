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

  bool runOnModule(Module &M) override {
     NamedMDNode *Annotations = M.getNamedMetadata("nvvm.annotations");
     LLVMContext& Context = M.getContext();
     Type* FloatPointerType = Type::getFloatPtrTy(Context);
     Type* Int32Type = Type::getInt32Ty(Context);
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
        Value *Output = Args--;
        
        
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
        Builder.CreateStore(ResultA, IntA);
        Value* ThreadIDX = XID->getPointerOperand();
        Value* Urem = Builder.CreateURem(
            Builder.CreateLoad(XID->getPointerOperand()), 
            Builder.CreateLoad(OriginalXAddr)
        );
        Builder.CreateStore(Urem, ThreadIDX );

        StoreInst* LastStore = getTheLastStore(NewKernelFunction);
        Value* CalculatedValue = LastStore->getValueOperand();
        GetElementPtrInst* Index = dyn_cast<GetElementPtrInst>(LastStore->getPointerOperand());
        Value* Pointer = dyn_cast<Value>(Index->idx_begin());
        errs() << *Pointer<< "\n";

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
        Builder.CreateStore(CalculatedValue, DA1Location);

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
        errs() << *NewKernelFunction << "\n";

      break;
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
  bool runOnModule(Module &M) override {

    CallInst* Cuda;
    LLVMContext& Context = M.getContext();
    Type* Int32Type = Type::getInt32Ty(Context);
    Type* VoidType = Type::getVoidTy(Context);
    for (Module::iterator F = M.begin(); F != M.end(); ++F) {
      for (Function::iterator BB = F->begin(); BB != F->end(); ++BB) {
        for (BasicBlock::iterator CurrentInstruction = BB->begin(); CurrentInstruction != BB->end(); ++CurrentInstruction) {  
          if (CallInst *FunctionCall = dyn_cast<CallInst>(CurrentInstruction)) {
               StringRef FunctionName = FunctionCall->getCalledFunction()->getName();
                if(isReplicate(FunctionCall)){
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
                    
                    FunctionCall->setCalledFunction(NewKernelAsCallee);
                    
            }else if (FunctionName == "cudaConfigureCall") {
              Cuda = FunctionCall;
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


