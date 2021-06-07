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
using namespace llvm;
#define ArgumanOrder                                                           \
  1 // Cuda Register Fonksiyonu çağrılırken 1 arguman fonksiyonu veriyor.
    // Gelecek Cuda versiyonlarında değişme ihtimaline karşı en üste tanımladık.
#define NumberOfReplication 3

namespace {
struct Hello : public FunctionPass {
  static char ID;
  Hello() : FunctionPass(ID) {}
  Value *numberOfgrid = nullptr;
  Value *numberOfThread = nullptr;
  Type *ty = nullptr;
  Type *ty2 = nullptr;
  int replicated = 0;
  Value *size;
  AllocaInst *AllocaInst1 = nullptr;
  AllocaInst *AllocaInst2 = nullptr;
  AllocaInst *AllocaInst3 = nullptr;
  AllocaInst *AllocaInst4 = nullptr;
  CallInst *printf = nullptr;

  bool runOnFunction(Function &F) override {
    /*
    bool print = false;
    errs() << "----------------\n";
    for (Function::iterator fi = F.begin(); fi != F.end(); ++fi) {
      for (BasicBlock::iterator bi = fi->begin(); bi != fi->end(); ++bi) {
          errs() << *bi << "\n";
        if (CallInst *callinstr = dyn_cast<CallInst>(bi)) {
          StringRef functionName = callinstr->getCalledFunction()->getName();
          if (functionName == "_ZN4dim3C2Ejjj") {
            if (this->numberOfgrid == nullptr) {
              this->numberOfgrid = callinstr->getArgOperand(1);
              continue;
            }
            if (this->numberOfThread == nullptr) {
               this->numberOfThread = callinstr->getArgOperand(1);
               this->ty  =
    StructType::create({Type::getInt64Ty(F.getContext()),Type::getInt32Ty(F.getContext())});
              continue;
            }
          }
          if (functionName == "cudaConfigureCall") {
            Value *operand = callinstr->getArgOperand(3);
            LoadInst *load = dyn_cast<LoadInst>(operand);
            GetElementPtrInst *gep =
                dyn_cast<GetElementPtrInst>((load->getOperand(0)));
            AllocaInst *al = dyn_cast<AllocaInst>((gep->getOperand(0)));

            FunctionCallee DimFunction =
                F.getParent()->getFunction("_ZN4dim3C2Ejjj");
            Type *structType = DimFunction.getFunctionType()
                                   ->getParamType(0)
                                   ->getPointerElementType();
          }

          if (functionName == "cudaMalloc") {
            // errs() << *callinstr << "\n";
            BitCastInst *bit = dyn_cast<BitCastInst>(callinstr->getOperand(0));
            this->size = callinstr->getOperand(1);
          }
          if (F.getName() == "__cuda_register_globals") {
            Function *RegisterFunction = callinstr->getCalledFunction();
            IRBuilder<> Builder(callinstr);

            Function *MajorityVotingFunction =
                F.getParent()->getFunction("majorityVoting");


            Value *t = callinstr->getArgOperand(1);
            Constant *gep = dyn_cast<Constant>(t);
            Value *name = Builder.CreateGlobalStringPtr("majorityVoting");
            if (print == false)
              Builder.CreateCall(
                  RegisterFunction,
                  {callinstr->getArgOperand(0),
                   Builder.CreateBitCast(
                       MajorityVotingFunction,
                       Type::getInt8PtrTy(F.getParent()->getContext())),
                   Builder.CreateGlobalStringPtr("majorityVoting"),
                   Builder.CreateGlobalStringPtr("majorityVoting"),
                   ConstantInt::get(llvm::Type::getInt32Ty(F.getContext()), -1),
                   ConstantPointerNull::get(Type::getInt8PtrTy(F.getContext())),
                   ConstantPointerNull::get(Type::getInt8PtrTy(F.getContext())),
                   ConstantPointerNull::get(Type::getInt8PtrTy(F.getContext())),
                   ConstantPointerNull::get(Type::getInt8PtrTy(F.getContext())),
                   ConstantPointerNull::get(
                       Type::getInt32PtrTy(F.getContext()))});
            return false;
          }
          if (functionName == "_Z9vectorAddiPf" && this->replicated == 0) {
            std::vector<Value *> Params;
            int argSize = callinstr->arg_size();
            LoadInst *output =
                dyn_cast<LoadInst>(callinstr->getArgOperand(argSize - 1));
            AllocaInst *oo = dyn_cast<AllocaInst>(output->getOperand(0));
            //
            LLVMContext &context =
                callinstr->getParent()->getParent()->getParent()->getContext();
            FunctionCallee dimFunction =
                F.getParent()->getFunction("_ZN4dim3C2Ejjj");

            Type *structType = dimFunction.getFunctionType()
                                   ->getParamType(0)
                                   ->getPointerElementType();
            FunctionCallee callConfFunction =
                F.getParent()->getFunction("cudaConfigureCall");
            // FunctionCallee kernel =
            // F.getParent()->getFunction("_Z6kernelPfS_S_S_i");
            Function *MajorityVotingFunction =
                F.getParent()->getFunction("majorityVoting");
            if (MajorityVotingFunction == nullptr) {

              FunctionCallee MajorityVotingCallee =
                  F.getParent()->getOrInsertFunction(
                      "majorityVoting", Type::getVoidTy(context),
                      Type::getFloatPtrTy(context),
                      Type::getFloatPtrTy(context),
                      Type::getFloatPtrTy(context),
                      Type::getFloatPtrTy(context), Type::getInt32Ty(context)

                  );

              MajorityVotingFunction =
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
                  BasicBlock::Create(context, "entry", MajorityVotingFunction);

              IRBuilder<> builder2(EntryBlock);

              MaybeAlign *align1 = new MaybeAlign(8);
              MaybeAlign *align2 = new MaybeAlign(4);

              Value *Aptr = builder2.CreateAlloca(Type::getFloatPtrTy(context),
                                                  nullptr, "A.addr");
              dyn_cast<AllocaInst>(Aptr)->setAlignment(*align1);

              Value *Bptr = builder2.CreateAlloca(Type::getFloatPtrTy(context),
                                                  nullptr, "B.addr");
              dyn_cast<AllocaInst>(Bptr)->setAlignment(*align1);

              Value *Cptr = builder2.CreateAlloca(Type::getFloatPtrTy(context),
                                                  nullptr, "C.addr");
              dyn_cast<AllocaInst>(Cptr)->setAlignment(*align1);

              Value *Outputptr = builder2.CreateAlloca(
                  Type::getFloatPtrTy(context), nullptr, "output.addr");
              dyn_cast<AllocaInst>(Outputptr)->setAlignment(*align1);

              Value *Sizeptr = builder2.CreateAlloca(Type::getInt32Ty(context),
                                                     nullptr, "size.addr");
              dyn_cast<AllocaInst>(Sizeptr)->setAlignment(*align2);

              StoreInst *StoreA = builder2.CreateStore(A, Aptr);
              dyn_cast<StoreInst>(StoreA)->setAlignment(*align1);

              Value *StoreB = builder2.CreateStore(B, Bptr);
              dyn_cast<StoreInst>(StoreB)->setAlignment(*align1);

              Value *StoreC = builder2.CreateStore(C, Cptr);
              dyn_cast<StoreInst>(StoreC)->setAlignment(*align1);

              Value *StoreOutput = builder2.CreateStore(Output, Outputptr);
              dyn_cast<StoreInst>(StoreOutput)->setAlignment(*align1);

              Value *StoreSize = builder2.CreateStore(Size, Sizeptr);
              dyn_cast<StoreInst>(StoreSize)->setAlignment(*align1);

              Value *zero =
                  ConstantInt::get(llvm::Type::getInt64Ty(F.getContext()), 0);
              Value *zero32 =
                  ConstantInt::get(llvm::Type::getInt32Ty(F.getContext()), 0);
              Value *one32 =
                  ConstantInt::get(llvm::Type::getInt32Ty(F.getContext()), 0);
              Value *eight =
                  ConstantInt::get(llvm::Type::getInt64Ty(F.getContext()), 8);
              Value *onalti =
                  ConstantInt::get(llvm::Type::getInt64Ty(F.getContext()), 16);
              Value *yirmidort =
                  ConstantInt::get(llvm::Type::getInt64Ty(F.getContext()), 24);
              Value *otuziki =
                  ConstantInt::get(llvm::Type::getInt64Ty(F.getContext()), 32);
              Value *four =
                  ConstantInt::get(llvm::Type::getInt64Ty(F.getContext()), 4);

              Value *CudaSetupArgument =
                  F.getParent()->getFunction("cudaSetupArgument");


              Value *PrintFunction =
                  F.getParent()->getFunction("printf");
              Value *CudaLaunch = F.getParent()->getFunction("cudaLaunch");

              Value *bitCasted =
                  builder2.CreateBitCast(Aptr, Type::getInt8PtrTy(context));
              Value *called = builder2.CreateCall(CudaSetupArgument,
                                                  {bitCasted, eight, zero});

              Value *cmp = builder2.CreateICmpEQ(called, zero32);

              builder2.CreateRetVoid();
              Instruction *ret = dyn_cast<Instruction>(cmp)->getNextNode();
              Instruction *split = SplitBlockAndInsertIfThen(cmp, ret, false);
              split->getParent()->setName("setup.next");

              builder2.SetInsertPoint(split);

              bitCasted =
                  builder2.CreateBitCast(Bptr, Type::getInt8PtrTy(context));

              called = builder2.CreateCall(CudaSetupArgument,
                                           {bitCasted, eight, eight});
              cmp = builder2.CreateICmpEQ(called, zero32);
              split = SplitBlockAndInsertIfThen(
                  cmp, dyn_cast<Instruction>(cmp)->getNextNode(), false);
              split->getParent()->setName("setup.next");
              builder2.SetInsertPoint(split);

              bitCasted =
                  builder2.CreateBitCast(Cptr, Type::getInt8PtrTy(context));
              called = builder2.CreateCall(CudaSetupArgument,
                                           {bitCasted, eight, onalti});
              cmp = builder2.CreateICmpEQ(called, zero32);
              split = SplitBlockAndInsertIfThen(
                  cmp, dyn_cast<Instruction>(cmp)->getNextNode(), false);
              split->getParent()->setName("setup.next");
              builder2.SetInsertPoint(split);

              bitCasted = builder2.CreateBitCast(Outputptr,
                                                 Type::getInt8PtrTy(context));
              called = builder2.CreateCall(CudaSetupArgument,
                                           {bitCasted, eight, yirmidort});
              cmp = builder2.CreateICmpEQ(called, zero32);
              split = SplitBlockAndInsertIfThen(
                  cmp, dyn_cast<Instruction>(cmp)->getNextNode(), false);
              split->getParent()->setName("setup.next");
              builder2.SetInsertPoint(split);

              bitCasted =
                  builder2.CreateBitCast(Sizeptr, Type::getInt8PtrTy(context));
              builder2.CreateCall(PrintFunction,
    {builder2.CreateGlobalStringPtr("--+++%d\n"),builder2.CreateLoad(Sizeptr)});
              called = builder2.CreateCall(CudaSetupArgument,
                                           {bitCasted, four, otuziki        });
              cmp = builder2.CreateICmpEQ(called, zero32);
              split = SplitBlockAndInsertIfThen(
                  cmp, dyn_cast<Instruction>(cmp)->getNextNode(), false);

              split->getParent()->setName("setup.next");
              builder2.SetInsertPoint(split);
              // errs() << * << "-------\n";

              builder2.CreateCall(
                  CudaLaunch,
                  {builder2.CreateBitCast(MajorityVotingFunction,
                                          Type::getInt8PtrTy(context))});
              // builder2.CreateCall(F.getParent()->getFunction("printf"),
              // {builder2.CreateGlobalStringPtr("oo ye\neee bu oldu\n")});
              //  errs() << *(CudaSetupArgument->getType()) << "-------\n";
            }
            // FunctionType* majorityFunctionType = FunctionType::get()
            Value *sOutput;
            Params.push_back(dyn_cast<Value>(oo));
            for (int i = 0; i < 3; i++) {
              BasicBlock *nextBB = callinstr->getParent()->getNextNode();
              Instruction *fInstr = dyn_cast<Instruction>(nextBB->begin());
              // builder.SetInsertPoint(fInstr);
              IRBuilder<> builder(fInstr);
              if (i < 2) {
                Function *callCudaMalloc =
                    F.getParent()->getFunction("cudaMalloc");
                sOutput = builder.CreateAlloca(oo->getAllocatedType(), nullptr,
                                               "d_A_2");
                builder.CreateBitCast(sOutput, Type::getInt8PtrTy(context));
                Value *nullF = ConstantPointerNull::get(
                    dyn_cast<PointerType>(oo->getAllocatedType()));
                StoreInst *sOutputStore =
                    dyn_cast<StoreInst>(builder.CreateStore(nullF, sOutput));

                Value *sOutputCasted = builder.CreateBitCast(
                    sOutputStore->getPointerOperand(),
                    Type::getInt8PtrTy(context)->getPointerTo());
                Params.push_back(sOutput);

                builder.CreateCall(callCudaMalloc, {sOutputCasted, this->size});
                // builder.CreateStore(sOutput, null);
              }

              FunctionCallee replicated =
                  F.getParent()->getFunction("_Z9vectorAddiPf");
              Type *ty = StructType::create(
                  {Type::getInt64Ty(context), Type::getInt32Ty(context)});
              MaybeAlign *align = new MaybeAlign(4);

              AllocaInst *allocaInstr =
    builder.CreateAlloca(structType->getScalarType());
              allocaInstr->setAlignment(*align);
              AllocaInst *allocaInstr2 =
    builder.CreateAlloca(structType->getScalarType());
              allocaInstr2->setAlignment(*align);
              AllocaInst *allocaInstr3 = builder.CreateAlloca(this->ty);
              allocaInstr3->setAlignment(*align);
              AllocaInst *allocaInstr4 = builder.CreateAlloca(this->ty);
              allocaInstr4->setAlignment(*align);

              Value *one =
                  ConstantInt::get(llvm::Type::getInt32Ty(F.getContext()), 1);
              Value *zero =
                  ConstantInt::get(llvm::Type::getInt32Ty(F.getContext()), 0);

              Value *zero64 =
                  ConstantInt::get(llvm::Type::getInt64Ty(F.getContext()), 0);
              Value *null = ConstantPointerNull::get(dyn_cast<PointerType>(
                  callConfFunction.getFunctionType()->getParamType(5)));
              Value *tw =
                  ConstantInt::get(llvm::Type::getInt64Ty(F.getContext()), 12);
              builder.CreateCall(dimFunction,
                                 {allocaInstr, this->numberOfgrid, one, one});
              builder.CreateCall(
                  dimFunction, {allocaInstr2, this->numberOfThread, one, one});
              Value *X = builder.CreateBitCast(dyn_cast<Value>(allocaInstr3),
                                               Type::getInt8PtrTy(context));
              Value *Y = builder.CreateBitCast(allocaInstr,
                                               Type::getInt8PtrTy(context));
              builder.CreateMemCpy(X, *align, Y, *align, tw);
              Value *Z = builder.CreateInBoundsGEP(allocaInstr3, {zero, zero});
              Value *LZ = builder.CreateLoad(Z);
              Value *Q = builder.CreateInBoundsGEP(allocaInstr3, {zero, one});
              Value *ZL = builder.CreateLoad(Q);
              Value *XY = builder.CreateBitCast(allocaInstr4,
                                                Type::getInt8PtrTy(context));
              Value *YX = builder.CreateBitCast(allocaInstr2,
                                                Type::getInt8PtrTy(context));
              builder.CreateMemCpy(XY, *align, YX, *align, tw);
              Value *ZQ = builder.CreateInBoundsGEP(allocaInstr4, {zero, zero});
              Value *lZQ = builder.CreateLoad(ZQ);
              Value *QZ = builder.CreateInBoundsGEP(allocaInstr4, {zero, one});
              Value *lQZ = builder.CreateLoad(QZ);
              Value *XX = dyn_cast<Value>(builder.CreateCall(
                  callConfFunction, {LZ, ZL, lZQ, lQZ, zero64, null}));
              Value *YY = builder.CreateICmpNE(XX, one);
              // errs() << *YY << "\n";
              Instruction *splitted = SplitBlockAndInsertIfThen(
                  YY, dyn_cast<Instruction>(YY)->getNextNode(), false);
              builder.SetInsertPoint(splitted);
              if (i < 2) {
                Value *param = builder.CreateLoad(sOutput);
                callinstr = builder.CreateCall(replicated, {one, param});
              } else {
                std::vector<Value *> Params2;
                for (int i = 0; i < Params.size(); i++) {
                  Value *sParam = Params.at(i);
                  Params2.push_back(builder.CreateLoad(sParam));
                }
                Value *sParam = Params.at(0);
                Value *fff = ConstantInt::get(
                    llvm::Type::getInt32Ty(F.getContext()), 15);
                Params2.push_back(builder.CreateLoad(sParam));
                Params2.push_back(fff);
                for (int i = 0; i < Params2.size(); i++){
                  Value *sParam = Params2.at(i);
                  errs() << *sParam << "\n";
                }
                callinstr = builder.CreateCall(MajorityVotingFunction, Params2);
              }
              this->replicated = 1;
              bi++;
              bi++;
              bi++;
              // print=false;
            }
          }
        }
      }
    }
    return false;*/
  }
}; // end of struct Hello

struct Hello2 : public ModulePass {
  static char ID;
  Hello2() : ModulePass(ID) {}

  bool runOnModule(Module &M) override {

    NamedMDNode *Annotations = M.getNamedMetadata("nvvm.annotations");
    errs() << Annotations << "\n";

    LLVMContext &C = M.getContext();
    MDNode *N = MDNode::get(C, MDString::get(C, "kernel"));
    MDNode *TempN = MDNode::get(C, ConstantAsMetadata::get(ConstantInt::get(
                                       llvm::Type::getInt32Ty(C), 1)));
    // MDNode* TempA = MDNode::get(C,
    // ValueAsMetadata::get(MajorityVotingFunction));
    MDNode *Con = MDNode::concatenate(N, TempN);
    // Con = MDNode::concatenate(Con, TempA);

    LLVMContext &Context = M.getContext();
    FunctionCallee MajorityVotingCallee = M.getOrInsertFunction(
        "majorityVoting15", Type::getVoidTy(Context),
        Type::getFloatPtrTy(Context), Type::getFloatPtrTy(Context),
        Type::getFloatPtrTy(Context), Type::getFloatPtrTy(Context),
        Type::getInt64Ty(Context)

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

    AllocaInst *printFAlloca = Builder.CreateAlloca(d->getScalarType());

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
    // errs() << *TID << "\n" << *(bitCast->getDestTy()) << "\n";
    // auto XX =  Builder.CreateInBoundsGEP(printFAlloca,
    // {ConstantInt::get(Type::getInt32Ty(Context),0),ConstantInt::get(Type::getInt32Ty(Context),0)});
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

/*    
        Builder.CreateCall(
            M.getFunction("vprintf"),
            {Builder.CreateGlobalStringPtr("CUDA'nın içindeyiz\n"),
             ConstantPointerNull::get(Type::getInt8PtrTy(M.getContext()))});
    */
    Annotations->addOperand(MDNode::concatenate(
        MDNode::get(C, ValueAsMetadata::get(MajorityVotingFunction)), Con));

    Builder.CreateRetVoid();
    return false;
  }

}; // end of struct Hello

struct Hello3 : public ModulePass {
  static char ID;
  Hello3() : ModulePass(ID) {}

  bool runOnModule(Module &M) override {

    std::vector<StringRef> FunctionsToReplicate;
    std::vector<Value *> CudaMallocSizes;
    std::vector<Value *> CudaMallocsOperands;
    std::vector<Value *> CudaMemcpyOperands;

    std::vector<Value *> GridOperand;
    std::vector<Value *> ThreadOperand;
    std::vector<Value *> GridOperandY;
    std::vector<Value *> ThreadOperandY;
    Function *CudaRegisterFunction = M.getFunction("__cuda_register_globals");
    Function *CudaRegisterFunction2 = M.getFunction("__cudaRegisterFunction");
    Function *CudaSetupArgument = M.getFunction("cudaSetupArgument");
    Function *CudaMalloc = M.getFunction("cudaMalloc");
    Function *CudaMemCpy = M.getFunction("cudaMemcpy");
    FunctionCallee DimentionFunction = M.getFunction("_ZN4dim3C2Ejjj");

    FunctionCallee CudaConfigureCall = M.getFunction("cudaConfigureCall");
    Function *CudaLaunch = M.getFunction("cudaLaunch");

    LLVMContext &Context = M.getContext();
    Type *Int64Type = Type::getInt64Ty(Context);
    Type *Int32Type = Type::getInt32Ty(Context);
    PointerType *Int8PtrType = Type::getInt8PtrTy(Context);
    PointerType *Int32PtrType = Type::getInt32PtrTy(Context);
    Type *CoercionType = StructType::create({Int64Type, Int32Type});
    Type *DimStructTypeScalar = DimentionFunction.getFunctionType()
                                    ->getParamType(0)
                                    ->getPointerElementType()
                                    ->getScalarType();

    PointerType *StreamType = dyn_cast<PointerType>(
        CudaConfigureCall.getFunctionType()->getParamType(5));
    Value *Zero32Bit = ConstantInt::get(Int32Type, 0);
    Value *Zero64Bit = ConstantInt::get(Int64Type, 0);
    Value *One32Bit = ConstantInt::get(Int32Type, 1);
    Value *Twelve64Bit = ConstantInt::get(Int64Type, 12);

    MaybeAlign *Align4 = new MaybeAlign(4);

    Value *StreamTypedNull = ConstantPointerNull::get(StreamType);
    StringRef CalledFunctionName;
    int DimensionFunction = 0;

    for (Function::iterator BB = CudaRegisterFunction->begin();
         BB != CudaRegisterFunction->end(); ++BB) {
      for (BasicBlock::iterator CurrentInstruction = BB->begin();
           CurrentInstruction != BB->end(); ++CurrentInstruction) {
        if (CallInst *CudaRegisterCall =
                dyn_cast<CallInst>(CurrentInstruction)) {
          Value *CudaFunctionOperand =
              CudaRegisterCall->getArgOperand(ArgumanOrder);
          Function *TestF =
              dyn_cast<Function>(CudaFunctionOperand->stripPointerCasts());
          StringRef FunctionName = TestF->getName();
          FunctionsToReplicate.push_back(FunctionName);
        }
      }
    }

    for (Module::iterator F = M.begin(); F != M.end(); ++F) {
      for (Function::iterator BB = F->begin(); BB != F->end(); ++BB) {
        for (BasicBlock::iterator CurrentInstruction = BB->begin();
             CurrentInstruction != BB->end(); ++CurrentInstruction) {

          if (CallInst *FunctionCall = dyn_cast<CallInst>(CurrentInstruction)) {
            CalledFunctionName = FunctionCall->getCalledFunction()->getName();
            std::vector<StringRef>::iterator Iterator =
                std::find(FunctionsToReplicate.begin(),
                          FunctionsToReplicate.end(), CalledFunctionName);
            if (CalledFunctionName == "cudaMalloc") {
              CudaMallocsOperands.push_back(FunctionCall->getArgOperand(0));
              CudaMallocSizes.push_back(FunctionCall->getArgOperand(1));
            } else if (CalledFunctionName == "_ZN4dim3C2Ejjj") {
              if (DimensionFunction % 2 == 1) {
                GridOperand.push_back(FunctionCall->getArgOperand(1));
                GridOperandY.push_back(FunctionCall->getArgOperand(2));
              } else {
                ThreadOperand.push_back(FunctionCall->getArgOperand(1));
                ThreadOperandY.push_back(FunctionCall->getArgOperand(2));
              }
              DimensionFunction++;
            } else if (CalledFunctionName == "cudaMemcpy") {
              CudaMemcpyOperands.push_back(FunctionCall->getArgOperand(1));

            } else if (Iterator != FunctionsToReplicate.end()) {
              std::vector<Value *> ReplicatedParameters;

              FunctionCallee ReplicatedFunction =
                  M.getFunction(CalledFunctionName);

              Value *CudaMallocSize;
              int ArgSize = FunctionCall->arg_size();
              LoadInst *Output =
                  dyn_cast<LoadInst>(FunctionCall->getArgOperand(ArgSize - 1));
              AllocaInst *OutputAllocation =
                  dyn_cast<AllocaInst>(Output->getOperand(0));
              int CountedIndex;
              for (auto &U : OutputAllocation->uses()) {
                User *User = U.getUser();
                if (BitCastInst *BitCast = dyn_cast<BitCastInst>(User)) {
                  std::vector<Value *>::iterator IteratorMalloc = std::find(
                      CudaMallocsOperands.begin(), CudaMallocsOperands.end(),
                      dyn_cast<Value>(User));
                  if (IteratorMalloc != CudaMallocsOperands.end()) {
                    CountedIndex = IteratorMalloc - CudaMallocsOperands.begin();
                    errs() << CountedIndex << "\n";
                    CudaMallocSize = CudaMallocSizes.at(CountedIndex);
                  }
                }
              }
              Type *MajorityVotingType = OutputAllocation->getAllocatedType();
              Type *OutputType = OutputAllocation->getAllocatedType();
              PointerType *OutputPtrType = dyn_cast<PointerType>(OutputType);

              Type *MajorityVotingPointerType = MajorityVotingType;

              Value *NullforOutputType =
                  ConstantPointerNull::get(OutputPtrType);
              int TypeID = MajorityVotingType->getTypeID();
              std::string MajorityVotingFunctionName =
                  "majorityVoting" + std::to_string(TypeID);
              Function *MajorityVotingFunction =
                  M.getFunction(MajorityVotingFunctionName);
              if (MajorityVotingFunction == nullptr) {
                FunctionCallee MajorityVotingCallee = M.getOrInsertFunction(
                    MajorityVotingFunctionName, Type::getVoidTy(Context),
                    MajorityVotingPointerType, MajorityVotingPointerType,
                    MajorityVotingPointerType, MajorityVotingPointerType,
                    Type::getInt64Ty(Context));
                std::vector<Value *> Parameters;

                MajorityVotingFunction =
                    dyn_cast<Function>(MajorityVotingCallee.getCallee());
                MajorityVotingFunction->setCallingConv(CallingConv::C);
                Function::arg_iterator Args =
                    MajorityVotingFunction->arg_begin();
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

                BasicBlock *EntryBlock = BasicBlock::Create(
                    Context, "entry", MajorityVotingFunction);

                IRBuilder<> Builder(EntryBlock);

                Builder.SetInsertPoint(EntryBlock);

                Value *Aptr = Builder.CreateAlloca(MajorityVotingPointerType,
                                                   nullptr, "A.addr");

                Value *Bptr = Builder.CreateAlloca(MajorityVotingPointerType,
                                                   nullptr, "B.addr");

                Value *Cptr = Builder.CreateAlloca(MajorityVotingPointerType,
                                                   nullptr, "C.addr");

                Value *Outputptr = Builder.CreateAlloca(
                    MajorityVotingPointerType, nullptr, "output.addr");

                Value *Sizeptr =
                    Builder.CreateAlloca(Int64Type, nullptr, "size.addr");

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
                for (unsigned long Index = 0; Index < Parameters.size();
                     Index++) {
                  Value *Parameter = Parameters.at(Index);
                  Value *BitcastParameter =
                      Builder.CreateBitCast(Parameter, Int8PtrType);
                  Value *OffsetValue = ConstantInt::get(Int64Type, Offset);
                  if (Parameter->getType() == Int32PtrType)
                    SizeParameter = 4;
                  else
                    SizeParameter =
                        8; // Diğerleri pointer olduğu için herhalde. Char*,
                           // float*, int* aynı çıktı.

                  Value *SizeValue = ConstantInt::get(Int64Type, SizeParameter);
                  Value *CudaSetupArgumentCall = Builder.CreateCall(
                      CudaSetupArgument,
                      {BitcastParameter, SizeValue, OffsetValue});
                  Instruction *IsError = dyn_cast<Instruction>(
                      Builder.CreateICmpEQ(CudaSetupArgumentCall, Zero32Bit));
                  if (Index == 0)
                    Builder
                        .CreateRetVoid(); // Buraya daha akıllıca çözüm bulmak
                                          // gerekiyor. Sevimli gözükmüyor.

                  Instruction *SplitPoint = SplitBlockAndInsertIfThen(
                      IsError, IsError->getNextNode(), false);

                  SplitPoint->getParent()->setName("setup.next");

                  Builder.SetInsertPoint(SplitPoint);
                  Offset += SizeParameter;
                }

                Builder.CreateCall(
                    CudaLaunch, {Builder.CreateBitCast(MajorityVotingFunction,
                                                       Int8PtrType)});

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
                     FunctionName, FunctionName,
                     ConstantInt::get(Int32Type, -1),
                     ConstantPointerNull::get(Int8PtrType),
                     ConstantPointerNull::get(Int8PtrType),
                     ConstantPointerNull::get(Int8PtrType),
                     ConstantPointerNull::get(Int8PtrType),
                     ConstantPointerNull::get(Int32PtrType)});
              }

              ReplicatedParameters.push_back(OutputAllocation);
              for (int Index = 0; Index < NumberOfReplication; Index++) {
                BasicBlock *CurrentBB = FunctionCall->getParent();
                BasicBlock *NextBB = CurrentBB->getNextNode();
                Instruction *FirstInstruction =
                    dyn_cast<Instruction>(NextBB->begin());
                IRBuilder<> Builder(FirstInstruction);

                std::vector<Value *> Arguments;
                int NumberArgument = FunctionCall->getNumArgOperands();

                for (int ArgumentIndex = 0; ArgumentIndex < NumberArgument - 1;
                     ArgumentIndex++) {
                  LoadInst *Argument = dyn_cast<LoadInst>(
                      FunctionCall->getArgOperand(ArgumentIndex));
                  Arguments.push_back(Argument->getOperand(0));

                  // Arguments.push_back(One32Bit);
                }
                Value *OutputReplication;
                if (Index < NumberOfReplication - 1) {
                  OutputReplication =
                      Builder.CreateAlloca(OutputType, nullptr, "dA_2");

                  Builder.CreateBitCast(OutputReplication, Int8PtrType);
                  StoreInst *OutputReplicationStore =
                      dyn_cast<StoreInst>(Builder.CreateStore(
                          NullforOutputType, OutputReplication));
                  Value *OutputReplicationCasted = Builder.CreateBitCast(
                      OutputReplicationStore->getPointerOperand(),
                      Int8PtrType->getPointerTo());
                  ReplicatedParameters.push_back(OutputReplication);
                  Builder.CreateCall(CudaMalloc,
                                     {OutputReplicationCasted, CudaMallocSize});
                  Arguments.push_back(OutputReplication);
                  Value *OutputReplicationLoad =
                      Builder.CreateLoad(OutputReplication);

                  // Value* innn = Builder.CreateCall(CudaMemCpy,
                  // {Builder.CreateBitCast(OutputReplicationLoad,Int8PtrType),
                  // CudaMemcpyOperands.at(CountedIndex),CudaMallocSize,
                  // One32Bit
                  // });
                }

                AllocaInst *Grid = Builder.CreateAlloca(DimStructTypeScalar);
                AllocaInst *Thread = Builder.CreateAlloca(DimStructTypeScalar);
                AllocaInst *GridCoercion = Builder.CreateAlloca(CoercionType);
                AllocaInst *ThreadCoercion = Builder.CreateAlloca(CoercionType);

                Builder.CreateCall(
                    DimentionFunction,
                    {Grid, GridOperand.at(GridOperand.size() - 1),
                     GridOperandY.at(GridOperand.size() - 1), One32Bit});
                Builder.CreateCall(
                    DimentionFunction,
                    {Thread, ThreadOperand.at(ThreadOperand.size() - 1),
                     ThreadOperandY.at(ThreadOperand.size() - 1), One32Bit});

                errs() << *ThreadOperand.at(ThreadOperand.size() - 1)
                       << "++**--\n";
                errs() << *GridOperand.at(GridOperand.size() - 1) << "++**--\n";

                Value *GridCoercionyBitCast = Builder.CreateBitCast(
                    dyn_cast<Value>(GridCoercion), Int8PtrType);
                Value *GridBitCast = Builder.CreateBitCast(Grid, Int8PtrType);

                Builder.CreateMemCpy(
                    GridCoercionyBitCast, *Align4, GridBitCast, *Align4,
                    Twelve64Bit); // Burada 12 olmasının sebebi: 64 bit ve 32
                                  // bitlik struct olması olabilir

                Value *XDimenGrid = Builder.CreateInBoundsGEP(
                    GridCoercion, {Zero32Bit, Zero32Bit});
                Value *LoadXDimenGrid = Builder.CreateLoad(XDimenGrid);
                Value *YDimenGrid = Builder.CreateInBoundsGEP(
                    GridCoercion, {Zero32Bit, One32Bit});
                Value *LoadYDimenGrid = Builder.CreateLoad(YDimenGrid);

                Value *ThreadCoercionBitCast =
                    Builder.CreateBitCast(ThreadCoercion, Int8PtrType);
                Value *ThreadBitCast =
                    Builder.CreateBitCast(Thread, Int8PtrType);

                Builder.CreateMemCpy(ThreadCoercionBitCast, *Align4,
                                     ThreadBitCast, *Align4, Twelve64Bit);

                Value *XDimenThread = Builder.CreateInBoundsGEP(
                    ThreadCoercion, {Zero32Bit, Zero32Bit});
                Value *LoadXDimenThread = Builder.CreateLoad(XDimenThread);
                Value *YDimenThread = Builder.CreateInBoundsGEP(
                    ThreadCoercion, {Zero32Bit, One32Bit});
                Value *LoadYDimenThread = Builder.CreateLoad(YDimenThread);
                /*
                    Builder.CreateCall(M.getFunction("printf"),
                                       {Builder.CreateGlobalStringPtr("%d
                   YOLOY\n"), LoadYDimenThread});

                    Builder.CreateCall(M.getFunction("printf"),
                                       {Builder.CreateGlobalStringPtr("%d
                   YOLOX\n"), LoadXDimenThread});*/
                Value *ConfigureCall = dyn_cast<Value>(Builder.CreateCall(
                    CudaConfigureCall,
                    {LoadXDimenGrid, LoadYDimenGrid, LoadXDimenThread,
                     LoadYDimenThread, Zero64Bit, StreamTypedNull}));

                Value *IsError = Builder.CreateICmpNE(ConfigureCall, One32Bit);

                Instruction *Splitted = SplitBlockAndInsertIfThen(
                    IsError, dyn_cast<Instruction>(IsError)->getNextNode(),
                    false);

                Builder.SetInsertPoint(Splitted);

                if (Index < NumberOfReplication - 1) {
                  std::vector<Value *> ReplicationFunctionArguments;
                  for (unsigned long Index2 = 0; Index2 < Arguments.size();
                       Index2++) {
                    Value *Parameter = Arguments.at(Index2);
                    ReplicationFunctionArguments.push_back(
                        Builder.CreateLoad(Parameter));
                  }
                  FunctionCall = Builder.CreateCall(
                      ReplicatedFunction, ReplicationFunctionArguments);
                } else {
                  std::vector<Value *> MajorityVotingArguments;
                  for (unsigned long Index2 = 0;
                       Index2 < ReplicatedParameters.size(); Index2++) {
                    Value *Parameter = ReplicatedParameters.at(Index2);
                    MajorityVotingArguments.push_back(
                        Builder.CreateLoad(Parameter));
                  }
                  Value *Parameter = ReplicatedParameters.at(0);
                  Value *Size = ConstantInt::get(Int64Type, 12);
                  MajorityVotingArguments.push_back(
                      Builder.CreateLoad(Parameter));
                  MajorityVotingArguments.push_back(CudaMallocSize);
                  FunctionCall = Builder.CreateCall(MajorityVotingFunction,
                                                    MajorityVotingArguments);
                }
              }

              FunctionsToReplicate.erase(Iterator);
            }
          }
        }
      }
    }

    return false;
  }

}; // end of struct Hello

struct Hello4 : public ModulePass {
  static char ID;
  Hello4() : ModulePass(ID) {}

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
    errs() << *Callee->getType() << "\n";
    errs() << *SecondReplicationCasted->getType() << "\n";
    errs() << *Size->getType() << "\n";
    errs() << *DestinationType->getPointerTo() << "\n";
    Builder.CreateCall(Callee, {SecondReplicationCasted, Size});
    // Builder.CreateLoad(Allocated);
    return Allocated;
  }

  std::pair<std::pair<std::vector<Value *>, std::vector<Value *>>,
            std::vector<Type *>>
  getDimensions(BasicBlock *BB) {
    StringRef GridDimName;
    StringRef BlockName;
    Value *GridDim;
    Value *BlockDim;

    std::vector<Value *> Block;
    std::vector<Value *> Grid;
    std::vector<Type *> Types;
    Type *DimensionType;
    Type *SteamType;
    std::vector<CallInst *> DimensionFunctions;
    for (BasicBlock::iterator CurrentInstruction = BB->begin();
         CurrentInstruction != BB->end(); ++CurrentInstruction) {
          // errs() << *CurrentInstruction << "\n";
      if (CallInst *FunctionCall = dyn_cast<CallInst>(CurrentInstruction)) {
        StringRef FunctionName = FunctionCall->getCalledFunction()->getName();
        StringRef DimensionVariableName;
        if (FunctionName == "cudaConfigureCall") {
          GridDim = dyn_cast<GetElementPtrInst>(
                        dyn_cast<LoadInst>(FunctionCall->getOperand(0))
                            ->getOperand(0))
                        ->getOperand(0);
          BlockDim = dyn_cast<GetElementPtrInst>(
                         dyn_cast<LoadInst>(FunctionCall->getOperand(2))
                             ->getOperand(0))
                         ->getOperand(0);
          /*
          SteamType = FunctionCall->getOperand(5)->getType();
          Types.push_back(SteamType);
          GridDimName = GridDim->getName().rtrim(".coerce");
          BlockName = BlockDim->getName().rtrim(".coerce");
          for (auto &DimensionCall : DimensionFunctions) {
            DimensionVariableName =
                DimensionCall->getOperand(0)->getValueName()->getKeyData();
            errs() << *DimensionCall << " " << DimensionVariableName << "++\n";
            if (DimensionVariableName == BlockName) {
              DimensionType = DimensionCall->getOperand(0)
                                  ->getType()
                                  ->getPointerElementType();
              Types.push_back(DimensionType);
              for (int Index = 1; Index < 4; Index++) {
                Block.push_back(DimensionCall->getOperand(Index));
              }
            } else if (DimensionVariableName == GridDimName) {
              for (int Index = 1; Index < 4; Index++) {
                Grid.push_back(DimensionCall->getOperand(Index));
              }
            }
          }*/
        } else if (FunctionName.contains("_ZN4dim3C2Ejjj") == true) {
          DimensionFunctions.push_back(FunctionCall);
        }
      }else if (MemCpyInst *MemoryCopy = dyn_cast<MemCpyInst>(CurrentInstruction)){
        errs() << *MemoryCopy << "+*+*\n";
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
    Value *Zero32Bit = ConstantInt::get(Int32Type, 0);
    Value *One32Bit = ConstantInt::get(Int32Type, 1);
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
        Builder.CreateInBoundsGEP(BlockCoercionAlloca, {Zero32Bit, Zero32Bit});
    Value *BlockXArg = Builder.CreateLoad(BlockX);
    Value *BlockY =
        Builder.CreateInBoundsGEP(BlockCoercionAlloca, {Zero32Bit, One32Bit});
    Value *BlockYArg = Builder.CreateLoad(BlockY);

    Value *GridCoercionBitcast =
        Builder.CreateBitCast(GridCoercionAlloca, Int8Ptr);
    Value *GridBitcast = Builder.CreateBitCast(GridAlloca, Int8Ptr);
    Builder.CreateMemCpy(GridCoercionBitcast, *Align4, GridBitcast, *Align4,
                         Twelve64Bit);
    Value *GridX =
        Builder.CreateInBoundsGEP(GridCoercionAlloca, {Zero32Bit, Zero32Bit});
    Value *GridXArg = Builder.CreateLoad(GridX);
    Value *GridY =
        Builder.CreateInBoundsGEP(GridCoercionAlloca, {Zero32Bit, One32Bit});
    Value *GridYArg = Builder.CreateLoad(GridY);
    Value *ConfigureCall =
        Builder.CreateCall(Configure, {BlockXArg, BlockYArg, GridXArg, GridYArg,
                                       Zero64Bit, NullSteam});
    Value *Condition = Builder.CreateICmpNE(ConfigureCall, One32Bit);
    Instruction *NewInstruction = SplitBlockAndInsertIfThen(
        Condition, dyn_cast<Instruction>(Condition)->getNextNode(), false);
    return NewInstruction;
  }

  Instruction *replicateTheFunction(IRBuilder<> Builder, CallInst *FunctionCall,
                                    std::vector<Value *> CreatedOutputs,
                                    std::vector<Value *> Args, int Z) {
    std::vector<Value *> Parameters;
      LLVMContext &Context =
          FunctionCall->getParent()->getParent()->getContext();
      Type *Int32Type = Type::getInt32Ty(Context);
      Value *One32Bit = ConstantInt::get(Int32Type, Z);
      Parameters.push_back(One32Bit);
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
    for (size_t Index = 1; Index < FunctionCall->arg_size(); Index++) {
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
    Function *CudaMalloc = M.getFunction("cudaMalloc");
    Function *DimFunction = M.getFunction("_ZN4dim3C2Ejjj");
    Function *ConfigureFunction = M.getFunction("cudaConfigureCall");
    CallInst *CUDA;
    LLVMContext &Context = M.getContext();
    Type *Int64Type = Type::getInt64Ty(Context);
    Type *Int32Type = Type::getInt32Ty(Context);
      Value *Zero32Bit = ConstantInt::get(Int32Type, 1);
    Type *CoercionType = StructType::create({Int64Type, Int32Type});
    std::vector<CallInst *> CudaMallocFunctionCalls;
    for (Module::iterator F = M.begin(); F != M.end(); ++F) {
      for (Function::iterator BB = F->begin(); BB != F->end(); ++BB) {
        for (BasicBlock::iterator CurrentInstruction = BB->begin();
             CurrentInstruction != BB->end(); ++CurrentInstruction) {   
          if (CallInst *FunctionCall = dyn_cast<CallInst>(CurrentInstruction)) {
            StringRef FunctionName =
                FunctionCall->getCalledFunction()->getName();
            if (isReplicate(FunctionCall)) {
              BasicBlock *CurrentBB = FunctionCall->getParent();
              BasicBlock *NextBB = CurrentBB->getNextNode();
              BasicBlock *PrevBB = CurrentBB->getPrevNode();
              Instruction *FirstInstruction =
                  dyn_cast<Instruction>(NextBB->begin());
              IRBuilder<> Builder(FirstInstruction);
              std::pair<Value *, std::pair<Type *, Type *>> SizeOfTheOutput;
              std::vector<std::vector<Value *>> MajorityVotingArgs;
              for (int i = 0; i < 2; i++) {
                MDNode *RedundancyMetadata =
                    FunctionCall->getMetadata("Redundancy");
                StringRef MetadataString =
                    getMetadataString(RedundancyMetadata);
                std::pair<std::vector<std::string>, std::vector<std::string>>
                    InputsAndOutputs = parseMetadataString(MetadataString);
                std::vector<std::string> Inputs = InputsAndOutputs.first;
                std::vector<std::string> Outputs = InputsAndOutputs.second;
                std::vector<Value *> CreatedOutputs;
                if (i != 2) {
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
                }
                errs() << *CUDA << "Bilmem ki tabiat\n";
                CUDA->getNumArgOperands();
                std::vector<Value *> Args1 ;
                for (int x = 0; x < CUDA->getNumArgOperands(); x++ ){
                  auto Arg = CUDA->getArgOperand(x);
                  Instruction* LoadInst = dyn_cast_or_null<Instruction>(Arg);
                  if(LoadInst != NULL){
                    Arg = Builder.CreateLoad(LoadInst ->getOperand(0));

                  }
                  Args1.push_back(Arg);
                }
                Instruction* NewInstruction = Builder.CreateCall(ConfigureFunction, Args1);
            Value *Condition = Builder.CreateICmpNE(NewInstruction, Zero32Bit);
             NewInstruction = SplitBlockAndInsertIfThen(
                Condition, dyn_cast<Instruction>(Condition)->getNextNode(), false);
                Builder.SetInsertPoint(NewInstruction);

                /*
                std::pair<std::pair<std::vector<Value *>, std::vector<Value *>>,
                          std::vector<Type *>>
                    DimensionsAndType = getDimensions(PrevBB);
                std::vector<Value *> Block = DimensionsAndType.first.first;
                std::vector<Value *> Grid = DimensionsAndType.first.second;
                std::vector<Type *> Types = DimensionsAndType.second;
                Instruction *NewInstruction = createDimensions(
                    ConfigureFunction, Builder, Block, Grid, Types.at(1),
                    CoercionType, DimFunction, Types.at(0));
                Builder.SetInsertPoint(NewInstruction);
                */
                
                std::vector<Value *> Args = getArgs(FunctionCall, Outputs);
                if (i != 2) {
                  Instruction *NewFunction = replicateTheFunction(
                      Builder, FunctionCall, CreatedOutputs, Args, i+1);
                  errs() << *NewFunction << "\n";
                  Instruction*  Inst = dyn_cast<Instruction>(
                      NewFunction->getParent()->getNextNode()->begin());
                  errs() << *Inst << "\n";

                  Builder.SetInsertPoint(dyn_cast<Instruction>(NewFunction->getParent()->getNextNode()->begin()));
                  MajorityVotingArgs.push_back(CreatedOutputs);
                } else {
                  /*
                  Type *TypeOfOutput = SizeOfTheOutput.second.first;
                  Function *F = M.getFunction("majorityVoting15");
                  if (F == nullptr)
                    F = createMajorityVoting(
                        M, dyn_cast<PointerType>(TypeOfOutput));
                  std::vector<Value *> Args;
                  // errs() <<
                  // *dyn_cast<Instruction>(FunctionCall->getArgOperand(0))->getOperand(0)
                  // << "\n";

                  Args.push_back(Builder.CreateLoad(
                      dyn_cast<Instruction>(FunctionCall->getArgOperand(1))
                          ->getOperand(0)));*/

                      /*
                  Args.push_back(
                      Builder.CreateLoad(MajorityVotingArgs.at(0).at(0)));
                  Args.push_back(
                      Builder.CreateLoad(MajorityVotingArgs.at(0).at(0)));
                  Args.push_back(
                      Builder.CreateLoad(MajorityVotingArgs.at(1).at(0)));
                  Args.push_back(
                      Builder.CreateLoad(MajorityVotingArgs.at(1).at(0)));
                  Args.push_back(Builder.CreateLoad(
                      dyn_cast<Instruction>(FunctionCall->getArgOperand(1))
                          ->getOperand(0)));

                  Type *Int32Type = Type::getInt64Ty(Context);
                  Value *Zero32Bit = ConstantInt::get(Int32Type, 15);
                  Args.push_back(Zero32Bit);
                  errs() << *F->getFunctionType() << "\n";
                  for(int i = 0; i < Args.size(); i++)
                  errs() << *Args.at(i)->getType() << "\n";
                  Builder.CreateCall(F, Args);
                  */
                }
              }
            } else if (FunctionName.contains("cudaMalloc")) {
              CudaMallocFunctionCalls.push_back(FunctionCall);
            }else if(FunctionName == "cudaConfigureCall"){
              CUDA = FunctionCall;
            }
          }
        }
      }
    }

    return false;
  }
};

} // end of anonymous namespace

char Hello::ID = -1;
char Hello2::ID = -2;
char Hello3::ID = -3;
char Hello4   ::ID = -4;

static RegisterPass<Hello> X("CUDA", "Hello World Pass", false, false);

static RegisterPass<Hello2> XX("CUDA2", "Hello World Pass", false, false);

static RegisterPass<Hello3> YXX("CUDA3", "Hello World Pass", false, false);

static RegisterPass<Hello4> YXXX("CUDA4", "Hello World Pass",
                                 false /* Only looks at CFG */,
                                 false /* Analysis Pass */);

static RegisterStandardPasses Y(PassManagerBuilder::EP_EarlyAsPossible,
                                [](const PassManagerBuilder &Builder,
                                   legacy::PassManagerBase &PM) {
                                  PM.add(new Hello());
                                });

static RegisterStandardPasses YX(PassManagerBuilder::EP_EarlyAsPossible,
                                 [](const PassManagerBuilder &Builder,
                                    legacy::PassManagerBase &PM) {
                                   PM.add(new Hello2());
                                 });

static RegisterStandardPasses YY(PassManagerBuilder::EP_EarlyAsPossible,
                                 [](const PassManagerBuilder &Builder,
                                    legacy::PassManagerBase &PM) {
                                   PM.add(new Hello3());
                                 });

static RegisterStandardPasses YYX(PassManagerBuilder::EP_EarlyAsPossible,
                                  [](const PassManagerBuilder &Builder,
                                     legacy::PassManagerBase &PM) {
                                    PM.add(new Hello4());
                                  });
