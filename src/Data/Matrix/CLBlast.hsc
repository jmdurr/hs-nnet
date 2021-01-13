{-# LANGUAGE ForeignFunctionInterface #-}
{-# LANGUAGE MultiParamTypeClasses #-}
module Data.Matrix.CLBlast where

import Foreign.C.Types
import Foreign.Ptr
import Data.Int
import Data.Matrix.OpenCL hiding (failLeft,clTry)
import Data.Word
import Foreign.ForeignPtr
import Foreign.Storable
import Foreign.Marshal.Alloc
import Data.Word
import Data.Vector.Storable
import Control.Monad.IO.Class
import Data.Proxy
import Debug.Trace

#include <clblasths.h> 
#include <openclhs.h>

newtype CLBlastStatusCode = CLBlastStatusCode CInt
    deriving (Eq,Show)
#{enum CLBlastStatusCode, CLBlastStatusCode, 
  CLBlastSuccess,
  CLBlastOpenCLCompilerNotAvailable,
  CLBlastTempBufferAllocFailure,
  CLBlastOpenCLOutOfResources,
  CLBlastOpenCLOutOfHostMemory,
  CLBlastOpenCLBuildProgramFailure,
  CLBlastInvalidValue,
  CLBlastInvalidCommandQueue,
  CLBlastInvalidMemObject,
  CLBlastInvalidBinary,
  CLBlastInvalidBuildOptions,
  CLBlastInvalidProgram,
  CLBlastInvalidProgramExecutable,
  CLBlastInvalidKernelName,
  CLBlastInvalidKernelDefinition,
  CLBlastInvalidKernel,
  CLBlastInvalidArgIndex,
  CLBlastInvalidArgValue,
  CLBlastInvalidArgSize,
  CLBlastInvalidKernelArgs,
  CLBlastInvalidLocalNumDimensions,
  CLBlastInvalidLocalThreadsTotal,
  CLBlastInvalidLocalThreadsDim,
  CLBlastInvalidGlobalOffset,
  CLBlastInvalidEventWaitList,
  CLBlastInvalidEvent,
  CLBlastInvalidOperation,
  CLBlastInvalidBufferSize,
  CLBlastInvalidGlobalWorkSize,
  CLBlastNotImplemented,
  CLBlastInvalidMatrixA,
  CLBlastInvalidMatrixB,
  CLBlastInvalidMatrixC,
  CLBlastInvalidVectorX,
  CLBlastInvalidVectorY,
  CLBlastInvalidDimension,
  CLBlastInvalidLeadDimA,
  CLBlastInvalidLeadDimB,
  CLBlastInvalidLeadDimC,
  CLBlastInvalidIncrementX,
  CLBlastInvalidIncrementY,
  CLBlastInsufficientMemoryA,
  CLBlastInsufficientMemoryB,
  CLBlastInsufficientMemoryC,
  CLBlastInsufficientMemoryX,
  CLBlastInsufficientMemoryY,
  CLBlastInsufficientMemoryTemp,
  CLBlastInvalidBatchCount,
  CLBlastInvalidOverrideKernel,
  CLBlastMissingOverrideParameter,
  CLBlastInvalidLocalMemUsage,
  CLBlastNoHalfPrecision,
  CLBlastNoDoublePrecision,
  CLBlastInvalidVectorScalar,
  CLBlastInsufficientMemoryScalar,
  CLBlastDatabaseError,
  CLBlastUnknownError,
  CLBlastUnexpectedError}


newtype CLBlastLayout = CLBlastLayout CInt deriving (Eq)
#{enum CLBlastLayout,CLBlastLayout,CLBlastLayoutRowMajor,CLBlastLayoutColMajor}

newtype CLBlastTranspose = CLBlastTranspose CInt deriving (Eq, Show)
#{enum CLBlastTranspose,CLBlastTranspose,CLBlastTransposeNo,CLBlastTransposeYes,CLBlastTransposeConjugate}

newtype CLBlastTriangle = CLBlastTriangle CInt deriving (Eq)
#{enum CLBlastTriangle, CLBlastTriangle, CLBlastTriangleUpper,CLBlastTriangleLower}

newtype CLBlastDiagonal = CLBlastDiagonal CInt deriving (Eq)
#{enum CLBlastDiagonal, CLBlastDiagonal, CLBlastDiagonalNonUnit, CLBlastDiagonalUnit}

newtype CLBlastSide = CLBlastSide CInt deriving (Eq)
#{enum CLBlastSide,CLBlastSide,CLBlastSideLeft,CLBlastSideRight}

newtype CLBlastKernelMode = CLBlastKernelMode CInt deriving (Eq)
#{enum CLBlastKernelMode, CLBlastKernelMode, CLBlastKernelModeConvolution, CLBlastKernelModeCrossCorrelation}

newtype CLBlastPrecision = CLBlastPrecision CInt deriving (Eq)
#{enum CLBlastPrecision, CLBlastPrecision, CLBlastPrecisionHalf, CLBlastPrecisionSingle, CLBlastPrecisionDouble, CLBlastPrecisionComplexDouble, CLBlastPrecisionComplexSingle}

type PtrSize = #type size_t

class (Storable a, Floating a) => CLBlastType a where
  clBlastTgemm :: CLBlastLayout -> CLBlastTranspose -> CLBlastTranspose -> PtrSize -> PtrSize -> PtrSize -> a -> CLMem_ -> PtrSize -> PtrSize -> CLMem_ -> PtrSize -> PtrSize -> a -> CLMem_ -> PtrSize -> PtrSize -> Ptr CLCommandQueue_ -> Ptr CLEvent_ -> IO CLBlastStatusCode
  clBlastTaxpy :: PtrSize -> a -> CLMem_ -> PtrSize -> PtrSize -> CLMem_ -> PtrSize -> PtrSize -> Ptr CLCommandQueue_ -> Ptr CLEvent_ -> IO CLBlastStatusCode
  clBlastTDot :: Proxy a -> PtrSize -> CLMem_ -> PtrSize -> CLMem_ -> PtrSize -> PtrSize -> CLMem_ -> PtrSize -> PtrSize -> Ptr CLCommandQueue_ -> Ptr CLEvent_ -> IO CLBlastStatusCode
  clBlastTCopy :: Proxy a -> PtrSize -> CLMem_ -> PtrSize -> PtrSize -> CLMem_ -> PtrSize -> PtrSize -> Ptr CLCommandQueue_ -> Ptr CLEvent_ -> IO CLBlastStatusCode
  toDouble :: a -> Double
  clTypeStringRep :: a -> String

instance CLBlastType CDouble where
  clBlastTgemm = clBlastDgemm
  clBlastTaxpy = clBlastDaxpy
  clBlastTDot _ = clBlastDdot
  clBlastTCopy _ = clBlastDCopy
  toDouble = fromRational . toRational
  clTypeStringRep _ = "double"

instance CLBlastType CFloat where
  clBlastTgemm = clBlastSgemm
  clBlastTaxpy = clBlastSaxpy
  clBlastTDot _ = clBlastSdot
  clBlastTCopy _ = clBlastSCopy
  toDouble = fromRational . toRational
  clTypeStringRep _ = "float"



failLeft :: (MonadFail m, MonadIO m) => IO (Either String a) -> m a
failLeft ie = do
  e <- liftIO ie
  case e of
    Left e' -> fail e'
    Right v -> pure v

clBlastTry :: (MonadIO m, MonadFail m) => IO CLBlastStatusCode -> IO a -> m (Either String a) 
clBlastTry st f = do
  r <- liftIO st
  if r /= clblastsuccess
    then pure (Left (show r))
    else liftIO $ Right <$> f

-- | clBlastGemm - dense matrix multiplication
clBlastGemm :: (MonadFail m, MonadIO m, CLBlastType a) => CLBlastLayout -> CLBlastTranspose -> CLBlastTranspose 
  -> Word64 -- ^ number of rows in first matrix
  -> Word64  -- ^ number of columns in second matrix
  -> Word64  -- ^ number of cols of first matrix and rows of second matrix
  -> a -- ^ scale first matrix by const
  -> CLMem a -- ^ first matrix memory
  -> Word64 -- ^  First matrix offset to start reading from
  -> Word64  -- ^ First matrix first dimension (rows for rowmajor, cols for colmajor)
  -> CLMem a -- ^ second matrix memory
  -> Word64 -- ^  Second matrix offset to start reading from
  -> Word64 -- ^ Second matrix first dimension (rows for rowmajor, cols for colmajor)
  -> a -- ^ constant to add to output matrix
  -> CLMem a -- ^ output matrix memory 
  -> Word64 -- ^ output matrix offset size
  -> Word64 -- ^ output matrix first dimension (rows for rowmajor, cols for colmajor)
  -> CLCommandQueue -> m CLEvent
clBlastGemm layout transA transB m n k alpha (CLMem al a ap) aoff lda (CLMem bl b bp) boff ldb beta (CLMem cl c cp) coff ldc queue =
  failLeft $
    alloca $ \e' ->
      clBlastTry 
        (withForeignPtr a (\a' ->
            withForeignPtr b (\b' -> 
              withForeignPtr c (\c' -> 
                withForeignPtr queue (\q' -> do
                    poke e' nullPtr
                    --putStrLn $ "  m:" <> show m <> " n:" <> show n <> " k:" <> show k <> " lda:" <> show lda <> " ldb:" <> show ldb <> " ldc: " <> show ldc
                    --putStrLn $ " transA:" <> show transA <> " transB:" <> show transB <> " aoff:" <> show aoff <> " boff:" <> show boff <> " coff:" <> show coff
                    clBlastTgemm layout transA transB m n k alpha a' aoff lda b' boff ldb beta c' coff ldc q' e'
                ) 
              )
            )
          )
        )
        (peek e' >>= clEventPtrToForeignPtr)

-- Vector * constant + Vector, overwrites the second vector
-- in future to avoid overwrite we copy the second vector to a new or existing third vector
clBlastAxpy :: (MonadFail m, MonadIO m, CLBlastType a) => CLCommandQueue -> CLMem a -> a -> CLMem a -> m CLEvent 
clBlastAxpy queue (CLMem len1 fpclmem1 prxy1) alpha (CLMem len2 fpclmem2 _)
  | len1 /= len2 = fail "Cannot add two vectors of different sizes"
  | otherwise =
    failLeft $ 
      alloca $ \e -> 
                  clBlastTry
                    (withForeignPtr queue (\queue' ->  
                        (withForeignPtr fpclmem1 (\mem1 -> 
                            withForeignPtr fpclmem2 (\mem2 -> do
                                poke e nullPtr
                                clBlastTaxpy (fromIntegral len1) alpha mem1 0 1 mem2 0 1 queue' e
                              )
                          )
                        )
                      )
                    )
                    (peek e >>= clEventPtrToForeignPtr)

clBlastDot :: (MonadFail m, MonadIO m, CLBlastType a) => CLCommandQueue -> CLMem a -> CLMem a -> CLMem a -> m CLEvent 
clBlastDot queue (CLMem sz1 fpclmem1 _) (CLMem sz2 fpclmem2 _) (CLMem sz3 fpclmem3 pxy)
  | sz1 /= sz2 = fail "Cannot dot two vectors of different sizes"
  | sz3 < 1 = fail "Must have at least one slot in result memory slot"
  | otherwise =
    failLeft $ 
      alloca $ \e -> 
                  clBlastTry
                    (withForeignPtr queue (\queue' ->  
                        (withForeignPtr fpclmem1 (\mem1 -> 
                            withForeignPtr fpclmem2 (\mem2 -> 
                              withForeignPtr fpclmem3 (\mem3 -> do
                                poke e nullPtr
                                clBlastTDot pxy (fromIntegral sz1) mem3 0 mem1 0 1 mem2 0 1 queue' e -- if this fails, need to call release on e
                              )
                            )
                          )
                        )
                      )
                    )
                    (peek e >>= clEventPtrToForeignPtr)
     

clBlastClone ::  (MonadFail m, MonadIO m, CLBlastType a) => CLContext -> CLCommandQueue -> CLMem a -> m (CLMem a)
clBlastClone ctx queue (CLMem sz fpmem pxy) = do
  (CLMem sz2 fpmem2 _) <- clCreateBuffer ctx clMemReadWrite sz pxy 
  failLeft $
    clBlastTry
      (alloca $ \e -> 
        (withForeignPtr queue (\queue' -> 
          (withForeignPtr fpmem (\mem -> 
            (withForeignPtr fpmem2 (\mem2 -> 
              do poke e nullPtr
                 clBlastTCopy pxy (fromIntegral sz) mem 0 1 mem2 0 1 queue' e 
              )
            )
          )
          )
        )
        )
      )
      (
        pure (CLMem sz2 fpmem2 pxy)
      )

foreign import ccall "CLBlastSaxpy" clBlastSaxpy :: PtrSize -> CFloat
    -> CLMem_ -> PtrSize -> PtrSize
    -> CLMem_ -> PtrSize -> PtrSize
    -> Ptr CLCommandQueue_ -> Ptr CLEvent_
    -> IO CLBlastStatusCode

foreign import ccall "CLBlastDaxpy" clBlastDaxpy :: PtrSize -> CDouble
    -> CLMem_ -> PtrSize -> PtrSize
    -> CLMem_ -> PtrSize -> PtrSize
    -> Ptr CLCommandQueue_ -> Ptr CLEvent_
    -> IO CLBlastStatusCode

-- alpha * A * x + beta * y
foreign import ccall "CLBlastDgemm" clBlastDgemm :: CLBlastLayout -> CLBlastTranspose -> CLBlastTranspose
    -> PtrSize -> PtrSize -> PtrSize -- Matrix A rows, Matrix A columns, 
    -> CDouble -- alpha (scalar alpha)
    -> CLMem_ -> PtrSize -> PtrSize -- mem, a offset, a ld
    -> CLMem_ -> PtrSize -> PtrSize -- mem b, b offset, b ld
    -> CDouble -- beta
    -> CLMem_ -> PtrSize -> PtrSize -- out buf, buf offset, buff ld
    -> Ptr CLCommandQueue_ -> Ptr CLEvent_
    -> IO CLBlastStatusCode

foreign import ccall "CLBlastSgemm" clBlastSgemm :: CLBlastLayout -> CLBlastTranspose -> CLBlastTranspose
    -> PtrSize -> PtrSize -> PtrSize -- Matrix A rows, Matrix A columns, 
    -> CFloat -- alpha (scalar alpha)
    -> CLMem_ -> PtrSize -> PtrSize -- mem, a offset, a ld
    -> CLMem_ -> PtrSize -> PtrSize -- mem b, b offset, b ld
    -> CFloat -- beta
    -> CLMem_ -> PtrSize -> PtrSize -- out buf, buf offset, buff ld
    -> Ptr CLCommandQueue_ -> Ptr CLEvent_
    -> IO CLBlastStatusCode

foreign import ccall "CLBlastDdot" clBlastDdot :: PtrSize -> CLMem_ -> PtrSize 
  -> CLMem_ -> PtrSize -> PtrSize
  -> CLMem_ -> PtrSize -> PtrSize
  -> Ptr CLCommandQueue_ -> Ptr CLEvent_
  -> IO CLBlastStatusCode

foreign import ccall "CLBlastSdot" clBlastSdot :: PtrSize -> CLMem_ -> PtrSize 
  -> CLMem_ -> PtrSize -> PtrSize
  -> CLMem_ -> PtrSize -> PtrSize
  -> Ptr CLCommandQueue_ -> Ptr CLEvent_
  -> IO CLBlastStatusCode

foreign import ccall "CLBlastScopy" clBlastSCopy :: PtrSize -> CLMem_ -> PtrSize -> PtrSize -> CLMem_ -> PtrSize -> PtrSize -> Ptr CLCommandQueue_ -> Ptr CLEvent_ -> IO CLBlastStatusCode
foreign import ccall "CLBlastDcopy" clBlastDCopy :: PtrSize -> CLMem_ -> PtrSize -> PtrSize -> CLMem_ -> PtrSize -> PtrSize -> Ptr CLCommandQueue_ -> Ptr CLEvent_ -> IO CLBlastStatusCode