{-# LANGUAGE ForeignFunctionInterface #-}
{-# LANGUAGE GADTs #-}
module Data.Matrix.OpenCL where


import Foreign.C.Types
import Data.Int
import Data.Word
import Foreign.Ptr
import Foreign.C.String
import Data.Bits
import Foreign.ForeignPtr (ForeignPtr(..),withForeignPtr, touchForeignPtr)
import Foreign.Concurrent (newForeignPtr)
import Data.Either
import Control.Monad.IO.Class
import Foreign.Marshal.Alloc
import Foreign.Marshal.Array
import Data.List (foldl1', intercalate)
import Foreign.Storable
import Data.Proxy
import Control.Monad (void)
import qualified Data.ByteString as BS
import Debug.Trace
import Data.Maybe (catMaybes)

#include <openclhs.h>

newtype CLDefine = CLDefine CInt deriving (Eq,Ord,Show)
instance Storable CLDefine where
  sizeOf _ = sizeOf (0 :: CInt)
  alignment _ = alignment (0 :: CInt)
  peek ptr = CLDefine <$> peek (castPtr ptr :: Ptr CInt)
  poke ptr (CLDefine v) = poke (castPtr ptr :: Ptr CInt) v


#{enum CLDefine, CLDefine, CL_SUCCESS,
    CL_DEVICE_NOT_FOUND,
    CL_DEVICE_NOT_AVAILABLE,
    CL_COMPILER_NOT_AVAILABLE,
    CL_MEM_OBJECT_ALLOCATION_FAILURE,
    CL_OUT_OF_RESOURCES,
    CL_OUT_OF_HOST_MEMORY,
    CL_PROFILING_INFO_NOT_AVAILABLE,
    CL_MEM_COPY_OVERLAP,
    CL_IMAGE_FORMAT_MISMATCH,
    CL_IMAGE_FORMAT_NOT_SUPPORTED,
    CL_BUILD_PROGRAM_FAILURE,
    CL_MAP_FAILURE,
    CL_MISALIGNED_SUB_BUFFER_OFFSET,
    CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST,
    CL_INVALID_VALUE,
    CL_INVALID_DEVICE_TYPE,
    CL_INVALID_PLATFORM,
    CL_INVALID_DEVICE,
    CL_INVALID_CONTEXT,
    CL_INVALID_QUEUE_PROPERTIES,
    CL_INVALID_COMMAND_QUEUE,
    CL_INVALID_HOST_PTR,
    CL_INVALID_MEM_OBJECT,
    CL_INVALID_IMAGE_FORMAT_DESCRIPTOR,
    CL_INVALID_IMAGE_SIZE,
    CL_INVALID_SAMPLER,
    CL_INVALID_BINARY,
    CL_INVALID_BUILD_OPTIONS,
    CL_INVALID_PROGRAM,
    CL_INVALID_PROGRAM_EXECUTABLE,
    CL_INVALID_KERNEL_NAME,
    CL_INVALID_KERNEL_DEFINITION,
    CL_INVALID_KERNEL,
    CL_INVALID_ARG_INDEX,
    CL_INVALID_ARG_VALUE,
    CL_INVALID_ARG_SIZE,
    CL_INVALID_KERNEL_ARGS,
    CL_INVALID_WORK_DIMENSION,
    CL_INVALID_WORK_GROUP_SIZE,
    CL_INVALID_WORK_ITEM_SIZE,
    CL_INVALID_GLOBAL_OFFSET,
    CL_INVALID_EVENT_WAIT_LIST,
    CL_INVALID_EVENT,
    CL_INVALID_OPERATION,
    CL_INVALID_GL_OBJECT,
    CL_INVALID_BUFFER_SIZE,
    CL_INVALID_MIP_LEVEL,
    CL_INVALID_GLOBAL_WORK_SIZE,
    CL_INVALID_PROPERTY,
    CL_PLATFORM_PROFILE,
    CL_PLATFORM_VERSION,
    CL_PLATFORM_NAME,
    CL_PLATFORM_VENDOR,
    CL_PLATFORM_EXTENSIONS,
    CL_NONE,
    CL_READ_ONLY_CACHE,
    CL_READ_WRITE_CACHE,
    CL_LOCAL,
    CL_GLOBAL,
    CL_EXEC_KERNEL,
    CL_EXEC_NATIVE_KERNEL,
    CL_CONTEXT_REFERENCE_COUNT,
    CL_CONTEXT_DEVICES,
    CL_CONTEXT_PROPERTIES,
    CL_CONTEXT_NUM_DEVICES,
    CL_CONTEXT_PLATFORM,
    CL_QUEUE_CONTEXT,
    CL_QUEUE_DEVICE,
    CL_QUEUE_REFERENCE_COUNT,
    CL_QUEUE_PROPERTIES,
    CL_R,
    CL_A,
    CL_RG,
    CL_RA,
    CL_RGB,
    CL_RGBA,
    CL_BGRA,
    CL_ARGB,
    CL_INTENSITY,
    CL_LUMINANCE,
    CL_Rx,
    CL_RGx,
    CL_RGBx,
    CL_SNORM_INT8,
    CL_SNORM_INT16,
    CL_UNORM_INT8,
    CL_UNORM_INT16,
    CL_UNORM_SHORT_565,
    CL_UNORM_SHORT_555,
    CL_UNORM_INT_101010,
    CL_SIGNED_INT8,
    CL_SIGNED_INT16,
    CL_SIGNED_INT32,
    CL_UNSIGNED_INT8,
    CL_UNSIGNED_INT16,
    CL_UNSIGNED_INT32,
    CL_HALF_FLOAT,
    CL_FLOAT,
    CL_MEM_OBJECT_BUFFER,
    CL_MEM_OBJECT_IMAGE2D,
    CL_MEM_OBJECT_IMAGE3D,
    CL_MEM_TYPE,
    CL_MEM_FLAGS,
    CL_MEM_SIZE,
    CL_MEM_HOST_PTR,
    CL_MEM_MAP_COUNT,
    CL_MEM_REFERENCE_COUNT,
    CL_MEM_CONTEXT,
    CL_MEM_ASSOCIATED_MEMOBJECT,
    CL_MEM_OFFSET,
    CL_IMAGE_FORMAT,
    CL_IMAGE_ELEMENT_SIZE,
    CL_IMAGE_ROW_PITCH,
    CL_IMAGE_SLICE_PITCH,
    CL_IMAGE_WIDTH,
    CL_IMAGE_HEIGHT,
    CL_IMAGE_DEPTH,
    CL_ADDRESS_NONE,
    CL_ADDRESS_CLAMP_TO_EDGE,
    CL_ADDRESS_CLAMP,
    CL_ADDRESS_REPEAT,
    CL_ADDRESS_MIRRORED_REPEAT,
    CL_FILTER_NEAREST,
    CL_FILTER_LINEAR,
    CL_SAMPLER_REFERENCE_COUNT,
    CL_SAMPLER_CONTEXT,
    CL_SAMPLER_NORMALIZED_COORDS,
    CL_SAMPLER_ADDRESSING_MODE,
    CL_SAMPLER_FILTER_MODE,
    CL_BUILD_SUCCESS,
    CL_BUILD_NONE,
    CL_BUILD_ERROR,
    CL_BUILD_IN_PROGRESS,
    CL_KERNEL_FUNCTION_NAME,
    CL_KERNEL_NUM_ARGS,
    CL_KERNEL_REFERENCE_COUNT,
    CL_KERNEL_CONTEXT,
    CL_KERNEL_PROGRAM,
    CL_KERNEL_WORK_GROUP_SIZE,
    CL_KERNEL_COMPILE_WORK_GROUP_SIZE,
    CL_KERNEL_LOCAL_MEM_SIZE,
    CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
    CL_KERNEL_PRIVATE_MEM_SIZE,
    CL_EVENT_COMMAND_QUEUE,
    CL_EVENT_COMMAND_TYPE,
    CL_EVENT_REFERENCE_COUNT,
    CL_EVENT_COMMAND_EXECUTION_STATUS,
    CL_EVENT_CONTEXT,
    CL_COMMAND_NDRANGE_KERNEL,
    CL_COMMAND_TASK,
    CL_COMMAND_NATIVE_KERNEL,
    CL_COMMAND_READ_BUFFER,
    CL_COMMAND_WRITE_BUFFER,
    CL_COMMAND_COPY_BUFFER,
    CL_COMMAND_READ_IMAGE,
    CL_COMMAND_WRITE_IMAGE,
    CL_COMMAND_COPY_IMAGE,
    CL_COMMAND_COPY_IMAGE_TO_BUFFER,
    CL_COMMAND_COPY_BUFFER_TO_IMAGE,
    CL_COMMAND_MAP_BUFFER,
    CL_COMMAND_MAP_IMAGE,
    CL_COMMAND_UNMAP_MEM_OBJECT,
    CL_COMMAND_MARKER,
    CL_COMMAND_ACQUIRE_GL_OBJECTS,
    CL_COMMAND_RELEASE_GL_OBJECTS,
    CL_COMMAND_READ_BUFFER_RECT,
    CL_COMMAND_WRITE_BUFFER_RECT,
    CL_COMMAND_COPY_BUFFER_RECT,
    CL_COMMAND_USER,
    CL_COMPLETE,
    CL_RUNNING,
    CL_SUBMITTED,
    CL_QUEUED,
    CL_BUFFER_CREATE_TYPE_REGION,
    CL_PROFILING_COMMAND_QUEUED,
    CL_PROFILING_COMMAND_SUBMIT,
    CL_PROFILING_COMMAND_START,
    CL_PROFILING_COMMAND_END,
    CL_KHRONOS_VENDOR_ID_CODEPLAY}


newtype CLMemFlag = CLMemFlag CInt deriving (Eq)
#{enum CLMemFlag,CLMemFlag,
  CL_MEM_READ_WRITE,
  CL_MEM_WRITE_ONLY,
  CL_MEM_READ_ONLY,
  CL_MEM_USE_HOST_PTR,
  CL_MEM_ALLOC_HOST_PTR,
  CL_MEM_COPY_HOST_PTR}

newtype CLDeviceType = CLDeviceType CInt deriving (Eq,Show)
#{enum CLDeviceType, CLDeviceType,
    CL_DEVICE_TYPE_DEFAULT,
    CL_DEVICE_TYPE_CPU,
    CL_DEVICE_TYPE_GPU,
    CL_DEVICE_TYPE_ACCELERATOR,
    CL_DEVICE_TYPE_ALL}

newtype CLCommandQueueProperty = CLCommandQueueProperty CLInt deriving (Eq)
#{enum CLCommandQueueProperty, CLCommandQueueProperty,
    CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE,
    CL_QUEUE_PROFILING_ENABLE}

newtype CLMapFlag = CLMapFlag CLInt deriving (Eq)
#{enum CLMapFlag, CLMapFlag, CL_MAP_READ, CL_MAP_WRITE}

newtype CLBool = CLBool CLInt deriving (Eq)
#{enum CLBool, CLBool, CL_TRUE, CL_FALSE}

memFlagFromList :: [CLMemFlag] -> CLMemFlag
memFlagFromList = foldr (\(CLMemFlag l) (CLMemFlag r) -> CLMemFlag (l .|. r)) (CLMemFlag 0)

propFlagFromList :: [CLCommandQueueProperty] -> CLCommandQueueProperty
propFlagFromList = foldr (\(CLCommandQueueProperty l) (CLCommandQueueProperty r) -> CLCommandQueueProperty (l .|. r)) (CLCommandQueueProperty 0)


type CLInt = #type cl_int
type CLULong = #type cl_ulong
type CLUInt = #type cl_uint
type CLPlatformID = Ptr ()
type CLDeviceID = Ptr ()
type CLContextProperties = Ptr ()
type CLMem_ = Ptr ()
data CLMem a = CLMem #{type size_t} (ForeignPtr ()) (Proxy a)
type CLContext = Ptr ()
type CLCommandQueue_ = Ptr ()
-- things can reallocate the queue (poo)
type CLCommandQueue = ForeignPtr CLCommandQueue_
type CLProgram = ForeignPtr ()
type CLProgram_ = Ptr ()
type CLKernel_ = Ptr ()
type CLKernel = ForeignPtr ()
type CLEvent_ = Ptr ()
type CLEvent = (ForeignPtr (), [ForeignPtr ()]) -- the event ptr and some touch foreignptr commands to keep memory alive...
type CLSampler = Ptr ()


clPlatformSize :: Int
clPlatformSize = #size cl_platform_id

failLeft :: (MonadFail m, MonadIO m) => IO (Either String v) -> m v
failLeft v = do
  e <- liftIO v
  either fail pure e

clTry :: IO CLDefine -> IO v -> IO (Either String v)
clTry rio vio = do
  r <- liftIO rio
  if r /= clSuccess 
    then pure $ Left (show r)
    else Right <$> vio

clNumPlatforms :: (MonadFail m, MonadIO m) => m CLUInt
clNumPlatforms = failLeft $ 
  alloca (\szPtr -> do
    clTry (clGetPlatformIDs 0 nullPtr szPtr)
          (peek szPtr)
  )

-- CLPlatformID should be in a foreign ptr
clGetPlatforms :: (MonadFail m, MonadIO m) => m [CLPlatformID]
clGetPlatforms = do
  n <- clNumPlatforms
  failLeft $ 
    allocaBytes (fromIntegral n * #{size cl_platform_id}) (\ptPtr -> do
      clTry (clGetPlatformIDs n ptPtr nullPtr)
            (peekArray (fromIntegral n) ptPtr)
    )
  
clNumDevices :: (MonadFail m, MonadIO m) => CLPlatformID -> m CLUInt
clNumDevices plt = failLeft $
  alloca (\ptr -> 
    clTry (clGetDeviceIDs plt clDeviceTypeAll 0 nullPtr ptr)
          (peek ptr)
  )

clGetDevices :: (MonadFail m, MonadIO m) => CLPlatformID -> m [CLDeviceID]
clGetDevices plt = do
  n <- clNumDevices plt
  failLeft $ 
    allocaBytes (fromIntegral n * #{size cl_device_id}) (\dPtr -> 
      clTry (clGetDeviceIDs plt clDeviceTypeAll n dPtr nullPtr)
            (peekArray (fromIntegral n) dPtr)
    )

clCreateContext :: (MonadFail m, MonadIO m) => [CLDeviceID] -> m CLContext
clCreateContext d = failLeft $
    allocaBytes (length d * #{size cl_device_id}) (\dPtr -> do
      pokeArray dPtr d
      alloca $ \nPtr -> do ctx <- clCreateContextInternal nullPtr 1 dPtr nullFunPtr nullPtr nPtr
                           clTry (peek nPtr) (pure ctx)
    )
  
clCreateCommandQueue :: (MonadFail m, MonadIO m) => CLContext -> CLDeviceID -> [CLCommandQueueProperty] -> m CLCommandQueue
clCreateCommandQueue ctx dev props =
  failLeft $ 
    alloca $ \ptr -> do
      q <- clCreateCommandQueue_ ctx dev (propFlagFromList props) ptr
      clTry (peek ptr) $ do
          ptr' <- malloc
          poke ptr' q
          newForeignPtr ptr' (freeCmdQueue ptr')
          
clCreateBuffer :: (Storable a, Floating a, MonadFail m, MonadIO m) => CLContext -> CLMemFlag -> Word64 -> Proxy a -> m (CLMem a)  
clCreateBuffer ctx flg len pxy = do
  --liftIO $ putStrLn $ "createBuf: " <> show len <> ":" <> show (sizeOf (asProxyTypeOf 0.0 pxy))
  failLeft $
    alloca $ \ptr -> do
      mem <- clCreateBuffer_ ctx flg (len * fromIntegral (sizeOf (asProxyTypeOf 0.0 pxy))) nullPtr ptr
      clTry (peek ptr) (newForeignPtr mem (clReleaseMemObject mem) >>= \v -> pure (CLMem len v pxy))


withBs :: [BS.ByteString] -> [CString] -> ([CString] -> IO a) -> IO a
withBs [] [] _ = fail "Cannot build program with no source code"
withBs [] cs iof = iof cs
withBs (b:bs) cs iof = BS.useAsCString b (\cs' -> withBs bs (cs ++ [cs']) iof)

clCreateProgramWithSource :: (MonadIO m, MonadFail m) => CLContext -> [BS.ByteString] -> m (CLProgram)
clCreateProgramWithSource ctx [] = fail "Cannot create an empty program"
clCreateProgramWithSource ctx xs = do -- wrap all of this in a big try catch (eww)
  --liftIO $ putStrLn "prog src"
  ptr <- liftIO $ alloca $ \e -> 
        withBs xs [] (\cs -> 
          withArray cs (\pcs ->
            do p <- clCreateProgramWithSource_ ctx (fromIntegral $ length xs) pcs nullPtr e
               r <- peek e
               if r /= clSuccess
                then fail ("Create program with source failed with code: " <> show r)
                else pure p
          )
        )
  liftIO $ newForeignPtr ptr (clReleaseProgram ptr)

newtype CLProgramBuildInfoFlag = CLProgramBuildInfoFlag CInt deriving (Eq,Show)

#{enum CLProgramBuildInfoFlag, CLProgramBuildInfoFlag, CL_PROGRAM_REFERENCE_COUNT,
    CL_PROGRAM_CONTEXT,
    CL_PROGRAM_NUM_DEVICES,
    CL_PROGRAM_DEVICES,
    CL_PROGRAM_SOURCE,
    CL_PROGRAM_BINARY_SIZES,
    CL_PROGRAM_BINARIES,
    CL_PROGRAM_BUILD_STATUS,
    CL_PROGRAM_BUILD_OPTIONS,
    CL_PROGRAM_BUILD_LOG}

clBuildProgram :: (MonadIO m, MonadFail m) => CLProgram -> [CLDeviceID] -> [String] -> m ()
clBuildProgram prog devs compileOpts =
  let opts = intercalate " " compileOpts
  in do
      --liftIO $ putStrLn "build prog"
      r <- liftIO $ 
        withArray devs (\pdev ->
              withForeignPtr prog $ \pprog -> 
                withCString opts (\popts -> clBuildProgram_ pprog (fromIntegral $ length devs) pdev popts nullPtr nullPtr)
            )
      if r == clSuccess
      then pure () 
      else do
        infos <- mapM (clGetProgramBuildInfo prog) devs
        fail $ "clBuildProgram failed with:\n" <> show infos
    
clCreateKernel :: (MonadIO m, MonadFail m) => CLProgram -> String -> m CLKernel
clCreateKernel prog name = do
  --liftIO $ putStrLn "create kern"
  kptr <- liftIO $ 
    withForeignPtr prog $ \pprog -> 
      withCString name $ \pname -> 
        alloca $ \e -> do
          k <- clCreateKernel_ pprog pname e
          r <- peek e
          if r /= clSuccess
            then fail $ "could not create kernel, err code: " <> show e
            else pure k
  liftIO $ newForeignPtr kptr (clReleaseKernel kptr)


clKernelFromSource :: (MonadIO m, MonadFail m) => CLContext -> [CLDeviceID] -> BS.ByteString -> String -> m (CLProgram, CLKernel)
clKernelFromSource ctx devs src fname =
  do --liftIO $ putStrLn "kern f src"
     prog <- clCreateProgramWithSource ctx [src]
     clBuildProgram prog devs []
     k <- clCreateKernel prog fname  
     pure (prog,k)


data CLProgramBuildInfo = CLProgramBuildInfo { clBuildStatus :: CLDefine
                                             , clBuildOptions :: String
                                             , clBuildLog :: String
} deriving (Eq,Show)

-- on info failure need to escape IO to call right fail
clGetProgramBuildInfo :: (MonadFail m, MonadIO m) => CLProgram -> CLDeviceID -> m CLProgramBuildInfo
clGetProgramBuildInfo prog dev = do
  --liftIO $ putStrLn "build info"
  liftIO $ 
    withForeignPtr prog $ \pprog -> 
      allocaBytes pSize $ \ptr -> 
        alloca $ \szPtr -> do
          infoErr $ clGetProgramBuildInfo_ pprog dev clProgramBuildStatus (fromIntegral pSize) ptr szPtr
          status <- peek (castPtr ptr :: Ptr CLDefine)
          infoErr $ clGetProgramBuildInfo_ pprog dev clProgramBuildOptions (fromIntegral pSize) ptr szPtr
          opts <- peekCString (castPtr ptr)
          infoErr $ clGetProgramBuildInfo_ pprog dev clProgramBuildLog (fromIntegral pSize) ptr szPtr
          log <- peekCString (castPtr ptr)
          pure $ CLProgramBuildInfo status opts log

  where pSize = 64000
        infoErr fio = do
          r <- fio
          if r /= clSuccess
            then fail "clGetProgramBuildInfo failed"
            else pure ()


clEnqueueWriteBuffer :: (Floating a, Storable a, MonadFail m, MonadIO m) => CLCommandQueue -> CLMem a -> [a] -> m ()
clEnqueueWriteBuffer queue (CLMem len mem pxy) xs = do
  --liftIO $ putStrLn "writebuf"
  failLeft $ 
    let xsLen = fromIntegral (length xs) in
      if xsLen /= len
        then pure (Left "trying to enqueuewritebuffer with list size that does not match memory size")
        else do
          let bsz = len * fromIntegral (sizeOf (asProxyTypeOf 0.0 pxy))
          allocaBytes (fromIntegral bsz) $ \ptr -> do
            pokeArray ptr xs
            clTry (withForeignPtr queue (\queue' -> do
                    queuePtr <- peek queue' 
                    withForeignPtr mem (\mem' -> 
                      clEnqueueWriteBuffer_ queuePtr mem' clTrue 0 (fromIntegral bsz) (castPtr ptr) 0 nullPtr nullPtr)))
                  (pure ())

clBufferFromList :: (Floating a, Storable a, MonadFail m, MonadIO m) => CLContext -> CLCommandQueue -> [a] -> m (CLMem a)
clBufferFromList ctx q ls = do
  --liftIO $ putStrLn $ "buf from l: " <> show (length ls + length ls) -- TODO clblast memorycheck is messed up?
  b <- clCreateBuffer ctx clMemReadWrite (fromIntegral $ length ls) (undefined :: Proxy a) 
  clEnqueueWriteBuffer q b ls
  pure b
  
clEnqueueReadBuffer :: (Floating a, Storable a, MonadFail m, MonadIO m) => CLCommandQueue -> CLMem a -> m [a]
clEnqueueReadBuffer queue (CLMem len mem pxy) = do
  --liftIO $ putStrLn "readbuf"
  failLeft $
          let bsz = fromIntegral len *  (sizeOf (asProxyTypeOf 0.0 pxy)) in
            allocaBytes bsz 
              (\buf ->
                clTry (withForeignPtr queue (\queue' -> do
                                                queuePtr <- peek queue'
                                                withForeignPtr mem (\memPtr -> clEnqueueReadBuffer_ queuePtr memPtr clTrue 0 (fromIntegral bsz) buf 0 nullPtr nullPtr)
                                            )
                      )
                      (peekArray (fromIntegral len) (castPtr buf))
              )
          
clDebugBuffer :: (Floating a, Storable a, MonadFail m, MonadIO m, Show a) => CLCommandQueue -> CLMem a -> String -> m ()
clDebugBuffer q mem@(CLMem len mem' pxy) t = do
  ls <- clEnqueueReadBuffer q mem
  liftIO $ putStrLn (t <> " - first entry of mem: " <> show mem' <> " is " <> show (head ls)) 

clEventPtrToForeignPtr :: [ForeignPtr ()] -> Ptr () -> IO (ForeignPtr (), [ForeignPtr ()])
clEventPtrToForeignPtr mems ptr = do
  p <- newForeignPtr  ptr (clReleaseEvent ptr)
  pure (p,mems)

clWaitForEvent :: (MonadIO m, MonadFail m) => CLEvent -> m ()
clWaitForEvent (ev,tch) = do
  --liftIO $ putStrLn "wait ev"
  liftIO $ withForeignPtr ev (\ev' ->
    alloca (\ptr -> do
      poke ptr ev'
      void $ clWaitForEvents_ 1 ptr    
    )
    )
  liftIO $ mapM_ touchForeignPtr tch

clWaitEvents :: (MonadIO m, MonadFail m) => [CLEvent] -> m ()
clWaitEvents = mapM_ clWaitForEvent



newtype CLDeviceFPConfig = CLDeviceFPConfig CInt deriving (Eq,Show)
#{enum CLDeviceFPConfig, CLDeviceFPConfig, CL_FP_DENORM, CL_FP_INF_NAN, CL_FP_ROUND_TO_NEAREST, CL_FP_ROUND_TO_ZERO,
    CL_FP_ROUND_TO_INF, CL_FP_FMA, CL_FP_SOFT_FLOAT}


newtype CLDeviceInfoFlag = CLDeviceInfoFlag CInt deriving (Eq,Show)
#{enum CLDeviceInfoFlag, CLDeviceInfoFlag, 
    CL_DEVICE_VENDOR_ID,
    CL_DEVICE_MAX_COMPUTE_UNITS,
    CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS,
    CL_DEVICE_MAX_WORK_GROUP_SIZE,
    CL_DEVICE_MAX_WORK_ITEM_SIZES,
    CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR,
    CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT,
    CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT,
    CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG,
    CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT,
    CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE,
    CL_DEVICE_MAX_CLOCK_FREQUENCY,
    CL_DEVICE_ADDRESS_BITS,
    CL_DEVICE_MAX_READ_IMAGE_ARGS,
    CL_DEVICE_MAX_WRITE_IMAGE_ARGS,
    CL_DEVICE_MAX_MEM_ALLOC_SIZE,
    CL_DEVICE_IMAGE2D_MAX_WIDTH,
    CL_DEVICE_IMAGE2D_MAX_HEIGHT,
    CL_DEVICE_IMAGE3D_MAX_WIDTH,
    CL_DEVICE_IMAGE3D_MAX_HEIGHT,
    CL_DEVICE_IMAGE3D_MAX_DEPTH,
    CL_DEVICE_IMAGE_SUPPORT,
    CL_DEVICE_MAX_PARAMETER_SIZE,
    CL_DEVICE_MAX_SAMPLERS,
    CL_DEVICE_MEM_BASE_ADDR_ALIGN,
    CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE,
    CL_DEVICE_SINGLE_FP_CONFIG,
    CL_DEVICE_GLOBAL_MEM_CACHE_TYPE,
    CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE,
    CL_DEVICE_GLOBAL_MEM_CACHE_SIZE,
    CL_DEVICE_GLOBAL_MEM_SIZE,
    CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE,
    CL_DEVICE_MAX_CONSTANT_ARGS,
    CL_DEVICE_LOCAL_MEM_TYPE,
    CL_DEVICE_LOCAL_MEM_SIZE,
    CL_DEVICE_ERROR_CORRECTION_SUPPORT,
    CL_DEVICE_PROFILING_TIMER_RESOLUTION,
    CL_DEVICE_ENDIAN_LITTLE,
    CL_DEVICE_AVAILABLE,
    CL_DEVICE_COMPILER_AVAILABLE,
    CL_DEVICE_EXECUTION_CAPABILITIES,
    CL_DEVICE_QUEUE_PROPERTIES,
    CL_DEVICE_NAME,
    CL_DEVICE_VENDOR,
    CL_DRIVER_VERSION,
    CL_DEVICE_PROFILE,
    CL_DEVICE_VERSION,
    CL_DEVICE_EXTENSIONS,
    CL_DEVICE_PLATFORM,
    CL_DEVICE_PREFERRED_VECTOR_WIDTH_HALF,
    CL_DEVICE_HOST_UNIFIED_MEMORY,
    CL_DEVICE_NATIVE_VECTOR_WIDTH_CHAR,
    CL_DEVICE_NATIVE_VECTOR_WIDTH_SHORT,
    CL_DEVICE_NATIVE_VECTOR_WIDTH_INT,
    CL_DEVICE_NATIVE_VECTOR_WIDTH_LONG,
    CL_DEVICE_NATIVE_VECTOR_WIDTH_FLOAT,
    CL_DEVICE_NATIVE_VECTOR_WIDTH_DOUBLE,
    CL_DEVICE_NATIVE_VECTOR_WIDTH_HALF,
    CL_DEVICE_OPENCL_C_VERSION,
    CL_DEVICE_TYPE}

data CLDeviceInfo = CLDeviceInfo {  clDeviceInfoAvailable :: Bool
                                  , clDeviceInfoCanCompile :: Bool
                                  , clDeviceInfoIsLittleEndian :: Bool
                                  , clDeviceInfoExtensions :: [String]
                                  , clDeviceInfoMemorySize :: CLULong
                                  , clDeviceInfoHalfFPConfig :: CLDeviceFPConfig
                                  , clDeviceInfoImageSupport :: Bool
                                  , clDeviceInfoMaxComputeUnits :: CLUInt
                                  , clDeviceInfoMaxClockFrequency :: CLUInt
                                  , clDeviceInfoMaxConstantBufferSize :: CLULong
                                  , clDeviceInfoMaxMemAllocSize :: CLULong
                                  , clDeviceInfoMaxParameterSize :: CLUInt
                                  , clDeviceInfoMaxSamplers :: CLUInt
                                  , clDeviceInfoMaxWorkGroupSize :: #type size_t
                                  , clDeviceInfoMaxWorkItemDims :: CLUInt
                                  , clDeviceInfoName :: String
                                  , clDeviceInfoType :: CLDeviceType
                                  , clDeviceInfoVendor :: String
                                  , clDeviceInfoVendorId :: CLUInt
                                  , clDeviceInfoVersion :: String
                                  , clDeviceInfoDriverVersion :: String
} deriving (Show)


clDeviceInfo :: (MonadFail m, MonadIO m) => CLDeviceID -> m CLDeviceInfo
clDeviceInfo dev =
    CLDeviceInfo <$> clBoolParam clDeviceAvailable
                 <*> clBoolParam clDeviceCompilerAvailable
                 <*> clBoolParam clDeviceEndianLittle
                 <*> clSpaceArray clDeviceExtensions
                 <*> clNum clDeviceGlobalMemSize #{size cl_ulong}
                 <*> (CLDeviceFPConfig <$> clNum clDeviceSingleFpConfig #{size cl_device_fp_config})
                 <*> clBoolParam clDeviceImageSupport
                 <*> clNum clDeviceMaxComputeUnits #{size cl_uint}
                 <*> clNum clDeviceMaxClockFrequency #{size cl_uint}
                 <*> clNum clDeviceMaxConstantBufferSize #{size cl_ulong}
                 <*> clNum clDeviceMaxMemAllocSize #{size cl_ulong}
                 <*> clNum clDeviceMaxParameterSize #{size size_t}
                 <*> clNum clDeviceMaxSamplers #{size cl_uint}
                 <*> clNum clDeviceMaxWorkGroupSize #{size size_t}
                 <*> clNum clDeviceMaxWorkItemDimensions #{size cl_uint}
                 <*> clString clDeviceName
                 <*> (CLDeviceType <$> clNum clDeviceType #{size cl_device_type})
                 <*> clString clDeviceVendor
                 <*> clNum clDeviceVendorId #{size cl_uint}
                 <*> clString clDeviceVersion
                 <*> clString clDriverVersion

  where clBoolParam :: (MonadFail m, MonadIO m) => CLDeviceInfoFlag -> m Bool
        clBoolParam p = clGetParam p #{size size_t} (\ptr _ -> do 
                                                        v <- peek (castPtr ptr :: Ptr #{type size_t})
                                                        pure $ if v > 0 then True else False    
                                                    )
        clSpaceArray p = words <$> clString p
        clString p = clGetParam p 2048 (\ptr sz -> do
                                              cs <- peekArray (fromIntegral sz) (castPtr ptr :: Ptr CChar)
                                              pure $ map castCCharToChar cs
                                       )
        clNum :: (MonadFail m, MonadIO m, Num a, Storable a) => CLDeviceInfoFlag -> Int -> m a
        clNum p sz = clGetParam p sz (\ptr _ -> peek (castPtr ptr))

        clGetParam :: (MonadFail m, MonadIO m) => CLDeviceInfoFlag -> Int -> (Ptr () -> Word64 -> IO b) -> m b
        clGetParam p sz pf = 
          failLeft $
            allocaBytes sz (\ptr -> 
              allocaBytes #{size size_t} (\rsz -> 
                clTry (clGetDeviceInfo dev p (fromIntegral sz) ptr rsz)
                      ( peek rsz >>= pf ptr)
              )
            )
{- TODO  memory needs to be managed better, events need to ensure CLAMem does not get freed during execution of some function -}
data CLArgument where
  CLAPlain :: (Storable a) => a -> CLArgument
  CLAMem :: (Storable a) => CLMem a -> CLArgument

clRunKernel :: (MonadFail m, MonadIO m) => CLCommandQueue -> CLKernel -> [CLArgument] -> (Word64,Maybe Word64, Maybe Word64) -> m CLEvent 
clRunKernel queue kern args dims = do
  --liftIO $ putStrLn "ka 1"
  mems <-  liftIO $ catMaybes <$> mapM setKernelArg (zip [0..] args)
  --liftIO $ putStrLn "ka e"
  liftIO $ do
    --putStrLn ("nd1 " <> show dims)  
    ev <- withForeignPtr queue $ \queue' -> 
      withForeignPtr kern $ \kern' -> 
        withArray (dimArr dims) $ \dimPtr -> 
          alloca $ \e -> do
            q <- peek queue'
            r <- clEnqueueNDRangeKernel_ q kern' (numDims dims) nullPtr dimPtr nullPtr 0 nullPtr e
            if r /= clSuccess
              then fail ("clEnqueueNDRangeKernel failed with code: " <> show r)
              else (peek e >>= clEventPtrToForeignPtr (kern:mems))
    --putStrLn "nd2"
    pure ev

  where numDims (_,Just _, Just _) = 3
        numDims (_,Just _, Nothing) = 2
        numDims (_,Nothing,_) = 1
        dimArr (x,Nothing,_) = [x]
        dimArr (x,Just y, Nothing) = [x,y]
        dimArr (x,Just y, Just z) = [x,y,z]
        setKernelArg (i,CLAPlain flt) = withForeignPtr kern (\kpt -> alloca (\p -> poke p flt >> clSetKernelArg kpt i (fromIntegral $ sizeOf flt) (castPtr p))) >> pure Nothing
        setKernelArg (i,CLAMem (CLMem len mpt pxy)) = do
          withForeignPtr kern $ \kpt' ->
            withForeignPtr mpt $ \mpt' -> 
              alloca $ \mptf -> do
                poke mptf mpt'
                clSetKernelArg kpt' i #{size cl_mem} (castPtr mptf)
          pure (Just mpt)

foreign import ccall "clGetPlatformIDs" clGetPlatformIDs :: CLUInt -> Ptr CLPlatformID -> Ptr CLUInt -> IO CLDefine
foreign import ccall "clGetDeviceIDs" clGetDeviceIDs :: CLPlatformID -> CLDeviceType -> CLUInt -> Ptr CLDeviceID -> Ptr CLUInt -> IO CLDefine
foreign import ccall "clCreateBuffer" clCreateBuffer_ :: CLContext -> CLMemFlag -> #{type size_t} -> Ptr () -> Ptr CLDefine -> IO CLMem_
foreign import ccall "clReleaseMemObject" clReleaseMemObject :: Ptr () -> IO ()
foreign import ccall "clCreateContext" clCreateContextInternal :: Ptr () -> CLUInt -> Ptr CLDeviceID 
  -> FunPtr (CString -> Ptr () -> #{type size_t} -> Ptr () -> IO (Ptr ())) 
  -> Ptr () -> Ptr CLDefine -> IO CLContext
foreign import ccall "clCreateCommandQueue" clCreateCommandQueue_ :: CLContext -> CLDeviceID -> CLCommandQueueProperty -> Ptr CLDefine -> IO CLCommandQueue_
foreign import ccall "clReleaseCommandQueue" clReleaseCommandQueue_ :: CLCommandQueue_ -> IO ()
foreign import ccall "clEnqueueWriteBuffer" clEnqueueWriteBuffer_ :: CLCommandQueue_ -> CLMem_ -> CLBool -> #{type size_t} -> #{type size_t} -> Ptr () -> CLUInt -> Ptr CLEvent -> Ptr CLEvent -> IO CLDefine
foreign import ccall "clReleaseEvent" clReleaseEvent :: Ptr () -> IO ()
foreign import ccall "clEnqueueReadBuffer" clEnqueueReadBuffer_ :: CLCommandQueue_ -> CLMem_ -> CLBool -> Word64 -> Word64 -> Ptr () -> CLUInt -> Ptr CLEvent_ -> Ptr CLEvent_ -> IO CLDefine
foreign import ccall "clWaitForEvents" clWaitForEvents_ :: CLUInt -> Ptr CLEvent_ -> IO CLDefine
foreign import ccall "clGetDeviceInfo" clGetDeviceInfo :: CLDeviceID -> CLDeviceInfoFlag -> #{type size_t} -> Ptr () -> Ptr (#{type size_t}) -> IO CLDefine
foreign import ccall "clCreateProgramWithSource" clCreateProgramWithSource_ :: CLContext -> CLUInt -> Ptr CString -> Ptr #{type size_t} -> Ptr CLDefine -> IO CLProgram_
foreign import ccall "clBuildProgram" clBuildProgram_ :: CLProgram_ -> CLUInt -> Ptr CLDeviceID -> CString -> Ptr () -> Ptr () -> IO CLDefine
foreign import ccall "clReleaseProgram" clReleaseProgram :: CLProgram_ -> IO ()
foreign import ccall "clReleaseKernel" clReleaseKernel :: CLKernel_ -> IO ()
foreign import ccall "clCreateKernel" clCreateKernel_ :: CLProgram_ -> CString -> Ptr CLDefine -> IO CLKernel_
foreign import ccall "clSetKernelArg" clSetKernelArg :: CLKernel_ -> CLUInt -> #{type size_t} -> Ptr () -> IO CLDefine
foreign import ccall "clEnqueueNDRangeKernel" clEnqueueNDRangeKernel_ :: CLCommandQueue_ -> CLKernel_ -> CLUInt -> Ptr #{type size_t} -> Ptr #{type size_t} -> Ptr #{type size_t} -> CLUInt -> Ptr CLEvent_ -> Ptr CLEvent_ -> IO CLDefine
foreign import ccall "clGetProgramBuildInfo" clGetProgramBuildInfo_ :: CLProgram_ -> CLDeviceID -> CLProgramBuildInfoFlag -> #{type size_t} -> Ptr () -> Ptr #{type size_t} -> IO CLDefine
-- setKernelArg
-- EnqueueNDRangeKernel

freeCmdQueue :: Ptr CLCommandQueue_ -> IO ()
freeCmdQueue ptr = do
  ptr' <- peek ptr
  --putStrLn "release queue"
  clReleaseCommandQueue_  ptr'
  free ptr