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
using namespace llvm;


namespace {


struct Device : public ModulePass {
  static char ID;
  Device() : ModulePass(ID) {}

  bool runOnModule(Module &M) override {

    /*
    Auxiliary* PassAuxiliary = (Auxiliary*)malloc(sizeof(Auxiliary*));
    LLVMContext& Context = M.getContext();
    PassAuxiliary->VoidType = Type::getVoidTy(Context);
    PassAuxiliary->Int32Type = Type::getInt32Ty(Context);
    PassAuxiliary->Int64Type = Type::getInt64Ty(Context);
    errs() << "Burada" << "\n";
    PassAuxiliary->BlockDimX = M.getFunction("llvm.nvvm.read.ptx.sreg.ntid.x");
    PassAuxiliary->BlockIDX = M.getFunction("llvm.nvvm.read.ptx.sreg.ctaid.x");
    PassAuxiliary->ThreadIDX = M.getFunction("llvm.nvvm.read.ptx.sreg.tid.x");



    NamedMDNode *Annotations = M.getNamedMetadata("nvvm.annotations");
    std::vector<Function *> ValidKernels = getValidKernels(Annotations);
    MDNode *KernelNode = MDNode::get(Context, MDString::get(Context, "kernel"));
    
    for (auto& Kernel : ValidKernels) {
      FunctionType* FuncType = Kernel->getFunctionType();
      unsigned int NumberOfParam = FuncType->getNumParams();
      Type* OutputType = FuncType->getParamType(NumberOfParam -1);
      PointerType* OutputPtrType = dyn_cast_or_null<PointerType>(OutputType);
      if(OutputPtrType == nullptr){
        continue;
      }

          
      std::string MajorityFunctionName = "majorityVoting" + std::to_string(OutputType->getPointerElementType()->getTypeID());
      
      if(M.getFunction(MajorityFunctionName) == nullptr) {
        Function* MajorityVotingFunction = createDeviceMajorityVotingFunction(M, PassAuxiliary, OutputPtrType, MajorityFunctionName);
        MDNode *TempN = MDNode::get(Context, ConstantAsMetadata::get(ConstantInt::get(PassAuxiliary->Int32Type, 1)));
        MDNode *Con = MDNode::concatenate(KernelNode, TempN);
        Annotations->addOperand(MDNode::concatenate(MDNode::get(Context, ValueAsMetadata::get(MajorityVotingFunction)), Con));
      }

    }
    */
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
    PassAuxiliary->Zero32Bit = ConstantInt::get(PassAuxiliary->Int32Type, 0);;
    PassAuxiliary->One32Bit = ConstantInt::get(PassAuxiliary->Int32Type, 1);
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


                 PassAuxiliary->OutputObject = OutputObject;
            

                FTGPGPUPass* MyPass = new FTGPGPUPass(FunctionCall, PassAuxiliary);
                
                if(Scheme == "MKE"){
                  MyPass->setPass(new MKE());
                }else if (Scheme == "MKES") {
                  MyPass->setPass(new MKE(true));
                  
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



static RegisterStandardPasses Y(PassManagerBuilder::EP_EarlyAsPossible,
                                [](const PassManagerBuilder &Builder,
                                   legacy::PassManagerBase &PM) {
                                  PM.add(new Device());
                                });

static RegisterStandardPasses YHost(PassManagerBuilder::EP_EarlyAsPossible,
                                [](const PassManagerBuilder &Builder,
                                   legacy::PassManagerBase &PM) {
                                  PM.add(new Host());
                                });

