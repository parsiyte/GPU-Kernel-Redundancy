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
    PassAuxiliary->Int32PtrType = PointerType::getInt32PtrTy(Context);

    PassAuxiliary->Zero32Bit = ConstantInt::get(PassAuxiliary->Int32Type, 0);
    PassAuxiliary->Zero32Bit = ConstantInt::get(PassAuxiliary->Int64Type, 0);
    PassAuxiliary->One32Bit = ConstantInt::get(PassAuxiliary->Int32Type, 1);
    PassAuxiliary->Two32Bit = ConstantInt::get(PassAuxiliary->Int32Type, 2);

   
    PassAuxiliary->CudaDimensionFunctions[0] = M.getOrInsertFunction("llvm.nvvm.read.ptx.sreg.ctaid.x", PassAuxiliary->Int32Type); //blockId.x
    PassAuxiliary->CudaDimensionFunctions[1] = M.getOrInsertFunction("llvm.nvvm.read.ptx.sreg.ctaid.y", PassAuxiliary->Int32Type); //blockId.y
    PassAuxiliary->CudaDimensionFunctions[2] = M.getOrInsertFunction("llvm.nvvm.read.ptx.sreg.tid.x", PassAuxiliary->Int32Type); //threadId.x
    PassAuxiliary->CudaDimensionFunctions[3] = M.getOrInsertFunction("llvm.nvvm.read.ptx.sreg.tid.y", PassAuxiliary->Int32Type); //threadId.y
    PassAuxiliary->CudaDimensionFunctions[4] = M.getOrInsertFunction("llvm.nvvm.read.ptx.sreg.ntid.x", PassAuxiliary->Int32Type); //blockDim.x
    PassAuxiliary->CudaDimensionFunctions[5] = M.getOrInsertFunction("llvm.nvvm.read.ptx.sreg.ntid.y", PassAuxiliary->Int32Type); //blockDim.y
    
    char Dimensions[2] = {'x', 'y'};
    char Baseds[2] = {'b', 't'};

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
      std::vector<Type *> SplitType;
      for(size_t ArgIndex = 0; ArgIndex < Kernel->arg_size(); ArgIndex++){
        RevisitedType.push_back(Kernel->getArg(ArgIndex)->getType());
        SplitType.push_back(Kernel->getArg(ArgIndex)->getType());

      }

      for(int ReplicationIndex = 0; ReplicationIndex < NumberOfReplication - 1; ReplicationIndex++){
        RevisitedType.push_back(OutputType);
      }

      RevisitedType.push_back(PassAuxiliary->Int32Type);
      SplitType.push_back(PassAuxiliary->Int32Type);
      SplitType.push_back(PassAuxiliary->Int32Type);

      FunctionType* RevisitedKernelType = FunctionType::get(PassAuxiliary->VoidType, RevisitedType, false);
      FunctionType* SplitKernelType = FunctionType::get(PassAuxiliary->VoidType, SplitType, false);
      for(int BasedIndex = 0; BasedIndex < 2; BasedIndex++){
        for(int DimensionIndex = 0; DimensionIndex < 2; DimensionIndex++){
          char Dimension = Dimensions[DimensionIndex];
          char Based = Baseds[BasedIndex];
          int TypeIndex = 2 * BasedIndex + DimensionIndex;
          std::string NewKernelFunctionName = FunctionName + RevisitedSuffix + Based + Dimension;
          
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
          

          //alterTheFunction(NewKernelFunction, PassAuxiliary, TypeIndex);


          Annotations->addOperand(MDNode::concatenate(MDNode::get(Context, ValueAsMetadata::get(NewKernelFunction)), Con));
       }
      }

          
      std::string DetectionFunctionName = "Detection" + std::to_string(OutputType->getPointerElementType()->getTypeID());
      
      if(M.getFunction(DetectionFunctionName) == nullptr) {
        Function* DetectionFunction = createDeviceDetectionFunction(M, PassAuxiliary, OutputPtrType, DetectionFunctionName);
        Annotations->addOperand(MDNode::concatenate(MDNode::get(Context, ValueAsMetadata::get(DetectionFunction)), Con));
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
    PassAuxiliary->Two32Bit = ConstantInt::get(PassAuxiliary->Int32Type, 2);
    PassAuxiliary->Three32Bit = ConstantInt::get(PassAuxiliary->Int32Type, 3);
    PassAuxiliary->Four64Bit = ConstantInt::get(PassAuxiliary->Int64Type, 4);
    PassAuxiliary->Eight64Bit = ConstantInt::get(PassAuxiliary->Int64Type, 8);
    PassAuxiliary->CudaStreamNonBlocking = PassAuxiliary->One32Bit;
    PassAuxiliary->MinusOne32Bit = ConstantInt::get(PassAuxiliary->Int32Type, -1);

    PassAuxiliary->CudaMemCopy =  M.getFunction("cudaMemcpy");
    PassAuxiliary->CudaMalloc =  M.getFunction("cudaMalloc");
    PassAuxiliary->CudaMemFree =  M.getFunction("cudaFree");
    PassAuxiliary->CudaGlobalRegisterFunction = M.getFunction("__cuda_register_globals"); 
    PassAuxiliary->CudaRegisterFunction = M.getFunction("__cudaRegisterFunction");
    PassAuxiliary->CudaSetupArgument =  M.getFunction("cudaSetupArgument");
    PassAuxiliary->CudaLaunch = M.getFunction("cudaLaunch");
    PassAuxiliary->CudaThreadSync = M.getOrInsertFunction("cudaThreadSynchronize",PassAuxiliary->VoidType);
    PassAuxiliary->Int8PtrNull = ConstantPointerNull::get(PassAuxiliary->Int8PtrType);
    PassAuxiliary->Int32PtrNull = ConstantPointerNull::get(PassAuxiliary->Int32PtrType);

    FunctionType* MallocFunctionType = FunctionType::get(PassAuxiliary->Int8PtrType, {PassAuxiliary->Int64Type});
    FunctionCallee MallocCalle = M.getOrInsertFunction("malloc", MallocFunctionType);
    Function* MallocFunction = dyn_cast<Function>(MallocCalle.getCallee());
    MallocFunction->setReturnDoesNotAlias();
    PassAuxiliary->MallocFunction = MallocFunction;
    
    PassAuxiliary->FreeFunction = M.getOrInsertFunction("free", PassAuxiliary->VoidType, PassAuxiliary->Int8PtrType);

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

                 createOrInsertDetectionFunction(M, OutputObject, PassAuxiliary);

                
                 CudaConfigure(PassAuxiliary->CudaConfigureCall, PassAuxiliary->CudaConfiguration);

                 PassAuxiliary->OutputObject = OutputObject;
                 char MemoryType = Scheme[0];
                 char Dimension = Scheme[1];
                 char Based = Scheme[2];
                 FTGPGPUPass* MyPass = new FTGPGPUPass(FunctionCall, PassAuxiliary);

                
                
                 if(Scheme == "MKE"){
                   MyPass->setPass(new MKE());
                 }else if (Scheme == "MKES") {
                   MyPass->setPass(new MKE(true));
                 }else if (Scheme.substr(3) == "SKE"){
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

