{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}

module ML.NNet.FullyConnectedSGD where

import Control.Monad.IO.Class
import Data.BlasM
import Data.Proxy
import Debug.Trace
import GHC.TypeLits
import ML.NNet
import System.Random

-- | Layer State
data FFN mx i o = FFN
  { ffnW :: Matrix mx i o 1, -- weights by outputs, rows = inputs, cols = outputs -- 128,1
    ffnB :: Vector mx o
  }

-- | Input State
type FFNIn mx i = Vector mx i

-- | Gradient State
data FFNGrad mx i o = FFNGrad
  { ffnWGrad :: Matrix mx i o 1,
    ffnBGrad :: Vector mx o
  }

forwardProp :: (KnownNat i, KnownNat o, BlasM m mx) => FFN mx i o -> Vector mx i -> m (Vector mx o, FFNIn mx i)
forwardProp ffn input = do
  mr <- ffnW ffn `dense` input -- outputs Mat i o 1 -> Mat 1 i 1 -> Mat o 1 1
  mr' <- ffnB ffn `add` mr -- Mat o 1 1
  pure (mr', input)

backwardProp :: (Monad m, BlasM m mx, KnownNat i, KnownNat o, Show mx) => FFN mx i o -> FFNIn mx i -> Vector mx o -> m (Vector mx i, FFNGrad mx i o)
backwardProp ffn ffnOldInput dSub = do
  mW <- outer dSub ffnOldInput
  -- nW <- subtractM (ffnW ffn) mW -- o rows, i columns for each input a column of weights
  dy <- denseT1 (ffnW ffn) dSub
  pure (dy, FFNGrad mW dSub)

avgGrad :: (Monad m, BlasM m mx, KnownNat i, KnownNat o, Show mx) => [FFNGrad mx i o] -> m (FFNGrad mx i o)
avgGrad [] = error "Cannot average empty gradient"
avgGrad (g : gs) = go g gs
  where
    go g' [] = do
      wg <- applyFunction (ffnWGrad g') (Div Value (Const (fromIntegral $ 1 + length gs)))
      bg <- applyFunction (ffnBGrad g') (Div Value (Const (fromIntegral $ 1 + length gs)))
      pure $ FFNGrad wg bg
    go g' (ng : gs') = do
      vw <- add (ffnWGrad g') (ffnWGrad ng)
      vb <- add (ffnBGrad g') (ffnBGrad ng)
      go (FFNGrad vw vb) gs'

fsgdUpdate :: (Monad m, BlasM m mx, KnownNat i, KnownNat o, Show mx, GradientMod m igm mod (FFNGrad mx i o)) => FFN mx i o -> FFNGrad mx i o -> igm -> Maybe mod -> m (FFN mx i o, mod)
fsgdUpdate (FFN w b) grad igm mod = do
  (FFNGrad wg bg, mod') <- modGradient igm mod grad
  w2 <- subtractM w wg
  b2 <- subtractM b bg
  pure (FFN w2 b2, mod')

{-
  (Maybe a -> Matrix mx w h d -> m (Matrix mx w h d, a))
  -- mod doesn't know matrix size until it is called

-}

-- | create a fully connected sgd network given a learning rate
-- m a wi hi di wo ho dpo
fullyConnectedSGD :: (KnownNat i, KnownNat o, Monad m, BlasM m a, RandomGen g, GradientMod m igm mod (FFNGrad a i o)) => Proxy i -> Proxy o -> Layer m a (FFN a i o) (FFNIn a i) (FFNGrad a i o) 1 i 1 1 o 1 igm mod g
fullyConnectedSGD nip nop = Layer forwardProp backwardProp avgGrad fsgdUpdate (fullyConnectedSGDInit nip nop)

fullyConnectedSGDInit :: (BlasM m mx, KnownNat i, KnownNat o, RandomGen g) => Proxy i -> Proxy o -> (g -> (Double, g)) -> g -> m (FFN mx i o, g)
fullyConnectedSGDInit nip nop f gen =
  let (nw, gen1) = netRandoms f gen (fromIntegral $ natVal nip * natVal nop)
      (nb, gen2) = netRandoms f gen1 (fromIntegral $ natVal nop)
   in do
        --liftIO $ putStrLn $ "mxFromList fc sgd init 1 - " <> show (length nw)
        nw' <- mxFromList nw nip nop (undefined :: Proxy 1)
        --liftIO $ putStrLn $ "mxFromList fc sgd init 2 - " <> show (length nb)
        nb' <- mxFromList nb (undefined :: Proxy 1) nop (undefined :: Proxy 1)
        --liftIO $ putStrLn "mxFromList fc sgd init e"
        pure (FFN nw' nb', gen2)

-- o x i doubles
-- o doubles
-- o x i doubles
-- o doubles
-- i doubles
