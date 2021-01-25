{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}

module ML.NNet.FullyConnectedSGD where

import Control.Monad.IO.Class
import Data.BlasM
import Data.Proxy
import Debug.Trace
import GHC.TypeLits
import ML.NNet
import ML.NNet.Init.RandomFun
import System.Random

-- | Layer State
data FFN mx i o gst gst2 = FFN
  { ffnW :: Matrix mx i o 1, -- weights by outputs, rows = inputs, cols = outputs -- 128,1
    ffnB :: Vector mx o,
    ffnWGst :: Maybe gst,
    ffnBGst :: Maybe gst2
  }

-- | Input State
type FFNIn mx i = Vector mx i

-- | Gradient State
data FFNGrad mx i o = FFNGrad
  { ffnWGrad :: Matrix mx i o 1,
    ffnBGrad :: Vector mx o
  }

forwardProp :: (KnownNat i, KnownNat o, BlasM m mx) => FFN mx i o mod mod2 -> Vector mx i -> m (Vector mx o, FFNIn mx i)
forwardProp ffn input = do
  mr <- ffnW ffn `dense` input -- outputs Mat i o 1 -> Mat 1 i 1 -> Mat o 1 1
  mr' <- ffnB ffn `add` mr -- Mat o 1 1
  pure (mr', input)

backwardProp :: (Monad m, BlasM m mx, KnownNat i, KnownNat o, Show mx) => FFN mx i o mod mod2 -> FFNIn mx i -> Vector mx o -> m (Vector mx i, FFNGrad mx i o)
backwardProp ffn ffnOldInput dSub = do
  mW <- outer dSub ffnOldInput
  -- nW <- subtractM (ffnW ffn) mW -- o rows, i columns for each input a column of weights
  dy <- denseT1 (ffnW ffn) dSub
  pure (dy, FFNGrad mW dSub)

avgGrad :: (Monad m, BlasM m mx, KnownNat i, KnownNat o, Show mx) => [FFNGrad mx i o] -> m (FFNGrad mx i o)
avgGrad [] = error "Cannot average empty gradient"
avgGrad gs = do
  wg <- cellAvgMxs (map ffnWGrad gs)
  bg <- cellAvgMxs (map ffnBGrad gs)
  pure (FFNGrad wg bg)

--     (lst -> gd -> GradientDescentMethod gmod -> Maybe gmod -> m (lst, gmod)) ->
--     (Maybe state -> Matrix mx w h d -> Matrix mx w h d -> m (Matrix mx w h d, state)) ->
-- don't know anything about mod... it is the internal gradient state
-- need to keep two of these, one for weights and one for biases
-- should carry it in FFN
fsgdUpdate :: (Monad m, BlasM m mx, KnownNat i, KnownNat o, Show mx, GradientDescentMethod m mx conf mod i o 1, GradientDescentMethod m mx conf mod2 1 o 1) => conf -> FFN mx i o mod mod2 -> FFNGrad mx i o -> m (FFN mx i o mod mod2)
fsgdUpdate conf ffn grad = do
  (w', wgst) <- updateWeights conf (ffnWGst ffn) (ffnW ffn) (ffnWGrad grad)
  (b', bgst) <- updateWeights conf (ffnBGst ffn) (ffnB ffn) (ffnBGrad grad)
  pure (FFN w' b' (Just wgst) (Just bgst))

{-
  (Maybe a -> Matrix mx w h d -> m (Matrix mx w h d, a))
  -- mod doesn't know matrix size until it is called

-}

-- | create a fully connected sgd network given a learning rate
-- m a wi hi di wo ho dpo
fullyConnectedSGD :: (KnownNat i, KnownNat o, Monad m, BlasM m a, RandomGen g, GradientDescentMethod m a conf mod i o 1, GradientDescentMethod m a conf mod2 1 o 1) => Proxy i -> Proxy o -> Layer m a (FFN a i o mod mod2) (FFNIn a i) (FFNGrad a i o) 1 i 1 1 o 1 conf (mod, mod2) g
fullyConnectedSGD nip nop = Layer forwardProp backwardProp avgGrad fsgdUpdate (fullyConnectedSGDInit nip nop)

fullyConnectedSGDInit :: (BlasM m mx, KnownNat i, KnownNat o, RandomGen g) => Proxy i -> Proxy o -> WeightInitializer g -> g -> m (FFN mx i o mod mod2, g)
fullyConnectedSGDInit nip nop f gen =
  let (nw, gen1) = netRandoms f gen (fromIntegral $ natVal nip * natVal nop) (fromIntegral $ natVal nip) (fromIntegral $ natVal nop)
   in --(nb, gen2) = netRandoms f gen1 (fromIntegral $ natVal nop)
      do
        --liftIO $ putStrLn $ "mxFromList fc sgd init 1 - " <> show (length nw)
        nw' <- mxFromList nw nip nop (undefined :: Proxy 1)
        --liftIO $ putStrLn $ "mxFromList fc sgd init 2 - " <> show (length nb)
        nb' <- konst 0.0
        --liftIO $ putStrLn "mxFromList fc sgd init e"
        pure (FFN nw' nb' Nothing Nothing, gen1)

-- o x i doubles
-- o doubles
-- o x i doubles
-- o doubles
-- i doubles
