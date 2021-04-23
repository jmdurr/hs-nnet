{-# LANGUAGE DataKinds #-}
{-# LANGUAGE ExplicitForAll #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE ScopedTypeVariables #-}

module Main where

import Control.Monad
import Control.Monad.IO.Class
import Control.Monad.State.Lazy
import Criterion
import Criterion.Main
import Criterion.Types
import Data.BlasM
import Data.Matrix.CLBlasM
import Data.Proxy
import qualified Data.Vector.Storable as V
import Foreign.C.Types
import GHC.TypeLits
import Test.QuickCheck

cacheAddLayer :: CLBlasState CFloat -> IO (CLBlasState CFloat)
cacheAddLayer st =
  snd
    <$> execClGpu
      st
      ( do
          mx1 <- mxFromVec (V.fromList [0]) (Proxy :: Proxy 1) (Proxy :: Proxy 1) (Proxy :: Proxy 1)
          void $ addToAllWithDepth mx1 mx1
      )

genMatrix :: forall mx w h d. (KnownNat w, KnownNat h, KnownNat d) => Proxy '(w, h, d) -> CLBlasState CFloat -> IO ((Matrix (CLBlasMx CFloat) w h d, CLBlasState CFloat))
genMatrix _ st = do
  -- generate two arbitrary lists
  ls1 <- liftIO $ generate (take (fromIntegral $ natVal (Proxy :: Proxy w) * natVal (Proxy :: Proxy h) * natVal (Proxy :: Proxy d)) <$> infiniteListOf (choose (-60.0, 60.0)))
  execClGpu st $ do
    mx1 <- mxFromVec (V.fromList ls1) (Proxy :: Proxy w) (Proxy :: Proxy h) (Proxy :: Proxy d)
    pure mx1

-- speed test convolution
data ConvolveData mx = ConvolveData (Matrix mx 32 32 64) (Matrix mx 5 5 2048)

genConvolveData st = do
  (mx1, st1) <- genMatrix (Proxy :: Proxy '(32, 32, 64)) st
  (mx2, st2) <- genMatrix (Proxy :: Proxy '(5, 5, 2048)) st1
  -- cache convolve
  (_, st3) <- execClGpu st2 $ void $ convolve mx1 mx2 (Proxy :: Proxy 32) (Proxy :: Proxy 1, Proxy :: Proxy 1) (Proxy :: Proxy 0, Proxy :: Proxy 0, Proxy :: Proxy 0, Proxy :: Proxy 0) (Proxy :: Proxy 0, Proxy :: Proxy 0) (Proxy :: Proxy 0, Proxy :: Proxy 0)
  pure (mx1, mx2, st3)

func = Div (Mul Value (Const 3.9)) (Const 3.0)

main :: IO ()
main =
  defaultMainWith
    (defaultConfig {reportFile = Just "addLayer.htm"})
    [ envWithCleanup
        ( do
            si <- prepCLGpu (Proxy :: Proxy CFloat)
            (mx1, s) <- genMatrix (Proxy :: Proxy '(32, 32, 32)) si
            (mx2, s1) <- genMatrix (Proxy :: Proxy '(1, 1, 32)) s
            (mx3, s1') <- genMatrix (Proxy :: Proxy '(32, 32, 32)) s1
            (mx4, s2) <- genMatrix (Proxy :: Proxy '(1024, 1024, 1)) s1'
            (mx5, s31) <- genMatrix (Proxy :: Proxy '(1, 1024, 1)) s2

            (mx6, s33) <- genMatrix (Proxy :: Proxy '(1024, 1, 1)) s31
            s32 <- cacheAddLayer s33
            (cin, cf, s3) <- genConvolveData s32
            (_, s4) <- execClGpu s3 $ void $ mult mx1 mx1
            (_, s5) <- execClGpu s4 $ void $ applyFunction mx2 func
            (_, s6) <- execClGpu s5 $ void $ multToAllWithDepth mx1 mx2
            (_, s7) <- execClGpu s6 $ void $ Data.BlasM.flip mx2 FlipBoth
            (_, s8) <- execClGpu s7 $ void $ sumLayers mx2
            (_, s9) <- execClGpu s8 $ void $ dense mx4 mx5
            (_, s10) <- execClGpu s9 $ void $ add mx4 mx4
            pure ((mx1, mx2, mx3, mx4, mx5, mx6, cin, cf), s10)
        )
        (pure $ pure ())
        ( \ ~((mx1, mx2, mx3, mx4, mx5, mx6, cin, cf), st) ->
            bgroup
              "main"
              [ bench "addLayer" $ nfIO $ execClGpu st $ addToAllWithDepth mx1 mx2,
                bench "convolve" $ nfIO $ execClGpu st $ convolve cin cf (Proxy :: Proxy 32) (Proxy :: Proxy 2, Proxy :: Proxy 2) (Proxy :: Proxy 2, Proxy :: Proxy 1, Proxy :: Proxy 2, Proxy :: Proxy 1) (Proxy :: Proxy 0, Proxy :: Proxy 0) (Proxy :: Proxy 0, Proxy :: Proxy 0),
                bench "multiply haddamard" $ nfIO $ void $ execClGpu st $ mult mx4 mx4,
                bench "apply function" $ nfIO $ execClGpu st $ applyFunction mx4 func,
                bench "multLayer" $ nfIO $ execClGpu st $ multToAllWithDepth mx1 mx2,
                bench "flip" $ nfIO $ execClGpu st $ Data.BlasM.flip mx1 FlipBoth,
                bench "sum" $ nfIO $ execClGpu st $ sumLayers mx1,
                bench "dense" $ nfIO $ execClGpu st $ dense mx4 mx5,
                bench "dense ffn mx * vec" $ nfIO $ execClGpu st $ dense mx4 mx5,
                bench "dense realistic" $ nfIO $ execClGpu st $ dense mx4 mx4,
                bench "add" $ nfIO $ execClGpu st $ add mx4 mx4,
                bench "outer" $ nfIO $ execClGpu st $ reshapeM mx5 >>= \mx5' -> outer mx5 mx5',
                bench "denseT1" $ nfIO $ execClGpu st $ denseT1 mx4 mx5,
                bench "denseT2 realistic" $ nfIO $ execClGpu st $ denseT2 mx4 mx6
              ]
        )
    ]

{-
 convolve 32x32x64 with 5x5x2048 with local -> 330ms
 convolve 32x32x64 with 5x5x2048 no local -> 20ms

-}
