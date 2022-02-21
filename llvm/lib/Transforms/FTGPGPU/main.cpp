#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"
#include <cstddef>
#include <cstdlib>
#include <string>

#include "MKE.cpp"
#include "SKE.cpp"
using namespace llvm;


namespace {


struct Device : public ModulePass {
  static char ID;
  Device() : ModulePass(ID) {}

  bool runOnModule(Module &M) override {
    Auxiliary* PassAuxiliary = (Auxiliary*)malloc(sizeof(Auxiliary));
    LLVMContext& Context = M.getContext();
    PassAuxiliary->VoidType = Type::getVoidTy(Context);
    PassAuxiliary->Int32Type = Type::getInt32Ty(Context);
    PassAuxiliary->Int64Type = Type::getInt64Ty(Context);

    PassAuxiliary->Zero32Bit = ConstantInt::get(PassAuxiliary->Int32Type, 0);
    PassAuxiliary->One32Bit = ConstantInt::get(PassAuxiliary->Int32Type, 1);
    PassAuxiliary->Two32Bit = ConstantInt::get(PassAuxiliary->Int32Type, 2);


    PassAuxiliary->BlockDimX = M.getFunction("llvm.nvvm.read.ptx.sreg.ntid.x");
    PassAuxiliary->BlockIDX = M.getFunction("llvm.nvvm.read.ptx.sreg.ctaid.x");
    PassAuxiliary->ThreadIDX = M.getFunction("llvm.nvvm.read.ptx.sreg.tid.x");


    NamedMDNode *Annotations = M.getNamedMetadata("nvvm.annotations");
    std::vector<Function *> ValidKernels = getValidKernels(Annotations);
    MDNode *KernelNode = MDNode::get(Context, MDString::get(Context, "kernel"));
    MDNode *TempN = MDNode::get(Context, ConstantAsMetadata::get(ConstantInt::get(PassAuxiliary->Int32Type, 1)));
    MDNode *Con = MDNode::concatenate(KernelNode, TempN);
    
    for (auto& Kernel : ValidKernels) {
      FunctionType* FuncType = Kernel->getFunctionType();
      unsigned int NumberOfParam = FuncType->getNumParams();
      Type* OutputType = FuncType->getParamType(NumberOfParam -1);
      PointerType* OutputPtrType = dyn_cast_or_null<PointerType>(OutputType);
      std::string FunctionName = Kernel->getName();
      if(OutputPtrType == nullptr){
        continue;
      }
      std::vector<Type *> RevisitedType;
      for(size_t ArgIndex = 0; ArgIndex < Kernel->arg_size(); ArgIndex++){
        RevisitedType.push_back(Kernel->getArg(ArgIndex)->getType());

      }

      for(int ReplicationIndex = 0; ReplicationIndex < NumberOfReplication - 1; ReplicationIndex++){
        RevisitedType.push_back(OutputType);
      }

      RevisitedType.push_back(PassAuxiliary->Int32Type);

      FunctionType* RevisitedKernelType = FunctionType::get(PassAuxiliary->VoidType, RevisitedType, false);

      std::string NewKernelFunctionName = FunctionName + RevisitedSuffix;
      
      FunctionCallee NewKernelAsCallee =  M.getOrInsertFunction(NewKernelFunctionName, RevisitedKernelType);
      Function* NewKernelFunction = dyn_cast<Function>(NewKernelAsCallee.getCallee());



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
      Value *OriginalBased = Args--;
      Value *SecondRedundantArg = Args--;   
      Value *FirstRedundantArg = Args--;
      Value *OriginalOutput = Args--;

      Value* Output = NewKernelFunction->getArg(Kernel->arg_size() - 1);
      User* FirstUser =  Output->uses().begin()->getUser();
      StoreInst* OutputStore = dyn_cast<StoreInst>(FirstUser);
      AllocaInst* OutputAllocation = dyn_cast<AllocaInst>(OutputStore->getPointerOperand());

      BasicBlock* FirstBB = dyn_cast<BasicBlock>(NewKernelFunction->begin());
      
      Instruction& LastInstruction = FirstBB->front();

      std::string ValueName = OriginalOutput->getName().str();
      FirstRedundantArg->setName(ValueName + "1");
      SecondRedundantArg->setName(ValueName + "2");
      OriginalBased->setName("OriginalBased");

      IRBuilder<> Builder(OutputAllocation->getNextNode());

      Instruction* FirstRedundant =  Builder.CreateAlloca(OutputType,nullptr, "FirstRedundant");  
      Instruction* SecondRedundant =  Builder.CreateAlloca(OutputType,nullptr, "SecondRedundant");
      Value* OriginalBaseddr = Builder.CreateAlloca(PassAuxiliary->Int32Type,nullptr, "OriginalBased.addr");
      Instruction* MetaOutput = Builder.CreateAlloca(OutputType,nullptr, "MetaOutput");  

      Builder.CreateStore(OriginalBased, OriginalBaseddr);
      Builder.CreateStore(FirstRedundantArg, FirstRedundant);
      Builder.CreateStore(SecondRedundantArg, SecondRedundant);

      OutputAllocation->replaceAllUsesWith(MetaOutput);
        

        Builder.CreateStore(Output,OutputAllocation );
        

        Instruction* BlockIdaddr =  Builder.CreateAlloca(PassAuxiliary->Int32Type,nullptr, "BlockIdaddr");
        Instruction* BlockIdaddr2 =  Builder.CreateAlloca(PassAuxiliary->Int32Type,nullptr, "BlockIdaddr2");
        Instruction* BlockIdaddrY =  Builder.CreateAlloca(PassAuxiliary->Int32Type,nullptr, "BlockIdaddrY");
        

        
  
        Value* BlockIDYCall;
         Value* BlockYID2 ;
        Value* BlockIDXCall = Builder.CreateCall(PassAuxiliary->BlockIDX);

    
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


         changeXID(NewKernelFunction,BlockIDXCall,BlockIdaddr2, Builder);


         
        Builder.SetInsertPoint(OutputStore->getNextNode());
        BlockID = Builder.CreateLoad(BlockIdaddr);
        Instruction* ZeroCmp = dyn_cast<Instruction>(Builder.CreateICmpEQ(BlockID, PassAuxiliary->Zero32Bit));
        Instruction *ThenTerm, *FirstElseIfCondTerm;
        SplitBlockAndInsertIfThenElse(ZeroCmp, ZeroCmp->getNextNode(), &ThenTerm, &FirstElseIfCondTerm); 
        Builder.SetInsertPoint(ThenTerm);
        Builder.CreateStore(Builder.CreateLoad(OutputAllocation), MetaOutput);


        Instruction *ElseIfTerm, *SecondElseTerm;
        Builder.SetInsertPoint(FirstElseIfCondTerm);
        BlockID = Builder.CreateLoad(BlockIdaddr);
        Instruction* OneCmp = dyn_cast<Instruction>(Builder.CreateICmpEQ(BlockID, PassAuxiliary->One32Bit));
        SplitBlockAndInsertIfThenElse(OneCmp, OneCmp->getNextNode(), &ElseIfTerm, &SecondElseTerm); 
        Builder.SetInsertPoint(ElseIfTerm);
        Builder.CreateStore(Builder.CreateLoad(FirstRedundant), MetaOutput);


        Builder.SetInsertPoint(SecondElseTerm);
        BlockID = Builder.CreateLoad(BlockIdaddr);
        Instruction* TwoCmp = dyn_cast<Instruction>(Builder.CreateICmpEQ(BlockID, PassAuxiliary->Two32Bit));
        Instruction* NewBranch  = SplitBlockAndInsertIfThen(TwoCmp, TwoCmp->getNextNode(), false);
        Builder.SetInsertPoint(NewBranch);
        Builder.CreateStore(Builder.CreateLoad(SecondRedundant), MetaOutput);

       Annotations->addOperand(MDNode::concatenate(MDNode::get(Context, ValueAsMetadata::get(NewKernelFunction)), Con));





          
      std::string MajorityFunctionName = "majorityVoting" + std::to_string(OutputType->getPointerElementType()->getTypeID());
      
      if(M.getFunction(MajorityFunctionName) == nullptr) {
        Function* MajorityVotingFunction = createDeviceMajorityVotingFunction(M, PassAuxiliary, OutputPtrType, MajorityFunctionName);
        Annotations->addOperand(MDNode::concatenate(MDNode::get(Context, ValueAsMetadata::get(MajorityVotingFunction)), Con));
      }

    }
    
    return false;
  }

};

struct Host : public ModulePass {
  static char ID;

  Host() : ModulePass(ID) {}



  bool runOnModule(Module &M) override {

    LLVMContext &Context = M.getContext();
    Auxiliary* PassAuxiliary = (Auxiliary*) malloc(sizeof(Auxiliary));
    PassAuxiliary->Int8PtrType = PointerType::getInt8PtrTy(Context);
    PassAuxiliary->Int32PtrType = PointerType::getInt32PtrTy(Context);
    PassAuxiliary->VoidType = Type::getVoidTy(Context);
    PassAuxiliary->Int32Type = Type::getInt32Ty(Context);
    PassAuxiliary->Int64Type = Type::getInt64Ty(Context);
    PassAuxiliary->Zero32Bit = ConstantInt::get(PassAuxiliary->Int32Type, 0);
    PassAuxiliary->One32Bit = ConstantInt::get(PassAuxiliary->Int32Type, 1);
    PassAuxiliary->Three32Bit = ConstantInt::get(PassAuxiliary->Int32Type, 3);
    PassAuxiliary->Four64Bit = ConstantInt::get(PassAuxiliary->Int64Type, 4);
    PassAuxiliary->Eight64Bit = ConstantInt::get(PassAuxiliary->Int64Type, 8);
    PassAuxiliary->CudaStreamNonBlocking = PassAuxiliary->One32Bit;
    PassAuxiliary->MinusOne32Bit = ConstantInt::get(PassAuxiliary->Int32Type, -1);

    PassAuxiliary->CudaMemCopy =  M.getFunction("cudaMemcpy");
    PassAuxiliary->CudaGlobalRegisterFunction = M.getFunction("__cuda_register_globals"); 
    PassAuxiliary->CudaRegisterFunction = M.getFunction("__cudaRegisterFunction");
    PassAuxiliary->CudaSetupArgument =  M.getFunction("cudaSetupArgument");
    PassAuxiliary->CudaLaunch = M.getFunction("cudaLaunch");
    PassAuxiliary->CudaThreadSync = M.getOrInsertFunction("cudaThreadSynchronize",PassAuxiliary->VoidType);
    PassAuxiliary->Int8PtrNull = ConstantPointerNull::get(PassAuxiliary->Int8PtrType);
    PassAuxiliary->Int32PtrNull = ConstantPointerNull::get(PassAuxiliary->Int32PtrType);

    std::vector<CallInst * > CudaMemCpyFunctions;
    std::vector<CallInst * > CudaMallocFunctions;
    for (Module::iterator F = M.begin(); F != M.end(); ++F) {
      for (Function::iterator BB = F->begin(); BB != F->end(); ++BB) {
        for (BasicBlock::iterator CurrentInstruction = BB->begin(); CurrentInstruction != BB->end(); ++CurrentInstruction) {
          if (CallInst *FunctionCall = dyn_cast<CallInst>(CurrentInstruction)){
              StringRef FunctionName = FunctionCall->getCalledFunction()->getName();
              if(isReplicate(FunctionCall)){
                 StringRef Metadata = getMetadataString(FunctionCall);  
                 std::vector<std::string> MetadataExtract = parseData(Metadata);
                 
                 std::string OutputName = MetadataExtract[1];
                 std::string Scheme = MetadataExtract[2];
                 
                 Output* OutputObject = (Output*)malloc (sizeof(Output));
                 
                 PassAuxiliary->CudaMallocFunction =  getTheMemoryFunction(CudaMallocFunctions, OutputName, OutputObject);
                 PassAuxiliary->CudaMemCpyFunction =  getTheMemoryFunction(CudaMemCpyFunctions, OutputName);

                 createOrInsertMajorityVotingFunction(M,OutputObject, PassAuxiliary);

                 CudaConfigure(PassAuxiliary->CudaConfigureCall, PassAuxiliary->CudaConfiguration);


                 PassAuxiliary->OutputObject = OutputObject;
                 char Dimension = Scheme[0];
                 char Based = Scheme[1];

                 errs() << Dimension << "\n";
                 errs() << Based << "\n";

                 
                 
            

                 FTGPGPUPass* MyPass = new FTGPGPUPass(FunctionCall, PassAuxiliary);
                
                 if(Scheme == "MKE"){
                   MyPass->setPass(new MKE());
                 }else if (Scheme == "MKES") {
                   MyPass->setPass(new MKE(true));
                 }else if (Scheme.substr(2) == "SKE"){
                  MyPass->setPass(new SKE(Dimension, Based));
                  CurrentInstruction++;
                 }
                 
                  

                MyPass->runThePass();
              }else if (FunctionName == "cudaConfigureCall") {
                PassAuxiliary->StreamType = FunctionCall->getArgOperand(StreamArgIndex)->getType();
                PassAuxiliary->StreamPointType = FunctionCall->getArgOperand(StreamArgIndex)->getType()->getPointerTo();
                PassAuxiliary->StreamCreateFunction = M.getOrInsertFunction(StreamCreateFunctionName, PassAuxiliary->Int32Type,  PassAuxiliary->StreamPointType, PassAuxiliary->Int32Type);
                PassAuxiliary->CudaConfigureCall = FunctionCall;
              }else if (FunctionName == "cudaMemcpy") {
              CudaMemCpyFunctions.push_back(FunctionCall);
              }else if (FunctionName.contains("cudaMalloc")) {
              CudaMallocFunctions.push_back(FunctionCall);
            }






          }
        }
      }
    }
  


  
    return true;
  }




};
} // namespace
char Device::ID = -1;
char Host::ID = -2;

static RegisterPass<Device> DeviceRegister("Device", "FTGPGPU Device Pass", false, false);
static RegisterPass<Host> HostRegister("Host", "FTGPGPU Host Pass", false, false);



static RegisterStandardPasses YDevice(PassManagerBuilder::EP_EarlyAsPossible,
                                [](const PassManagerBuilder &Builder,
                                   legacy::PassManagerBase &PM) {
                                  PM.add(new Device());
                                });

static RegisterStandardPasses YHost(PassManagerBuilder::EP_EarlyAsPossible,
                                [](const PassManagerBuilder &Builder,
                                   legacy::PassManagerBase &PM) {
                                  PM.add(new Host());
                                });

