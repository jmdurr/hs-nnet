{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE ScopedTypeVariables #-}

module ML.NNet.FullyConnectedSGD where

import Control.Monad.IO.Class (liftIO)
import Data.BlasM
import Data.Proxy
import Data.Serialize
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

avgGrad :: (Monad m, BlasM m mx, KnownNat i, KnownNat o, Show mx) => (FFNGrad mx i o -> FFNGrad mx i o -> m (FFNGrad mx i o), FFNGrad mx i o -> Int -> m (FFNGrad mx i o))
avgGrad =
  ( \(FFNGrad w b) (FFNGrad w2 b2) -> FFNGrad <$> add w w2 <*> add b b2,
    \(FFNGrad w b) i -> FFNGrad <$> scale w (1.0 / fromIntegral i) <*> scale b (1.0 / fromIntegral i)
  )

--     (lst -> gd -> GradientDescentMethod gmod -> Maybe gmod -> m (lst, gmod)) ->
--     (Maybe state -> Matrix mx w h d -> Matrix mx w h d -> m (Matrix mx w h d, state)) ->
-- don't know anything about mod... it is the internal gradient state
-- need to keep two of these, one for weights and one for biases
-- should carry it in FFN
fsgdUpdate :: (Monad m, BlasM m mx, KnownNat i, KnownNat o, Show mx, GradientDescentMethod m mx conf mod i o 1, GradientDescentMethod m mx conf mod2 1 o 1) => conf -> FFN mx i o mod mod2 -> FFNGrad mx i o -> m (FFN mx i o mod mod2)
fsgdUpdate conf ffn grad = do
  (w', wgst) <- updateWeights conf (ffnWGst ffn) (ffnW ffn) (ffnWGrad grad)
  -- dwgt <- avgMxs [ffnW ffn]
  -- dwgt' <- avgMxs [w']
  -- liftIO $ putStrLn ("updated fsgd weights : " <> show dwgt <> " to " <> show dwgt')
  (b', bgst) <- updateWeights conf (ffnBGst ffn) (ffnB ffn) (ffnBGrad grad)
  pure (FFN w' b' (Just wgst) (Just bgst))

-- | Serialize and deserialize learning state
fsgdSerialize :: forall m mx i o conf mod mod2. (BlasM m mx, KnownNat i, KnownNat o, GradientDescentMethod m mx conf mod i o 1, GradientDescentMethod m mx conf mod2 1 o 1) => conf -> (Get (m (FFN mx i o mod mod2)), FFN mx i o mod mod2 -> m Put)
fsgdSerialize _ =
  ( do
      w <- deserializeMx (Proxy :: Proxy '(i, o, 1))
      b <- deserializeMx (Proxy :: Proxy '(1, o, 1))
      wg <- fst (serializeMod (Proxy :: Proxy '(conf, i, o, 1)))
      bg <- fst (serializeMod (Proxy :: Proxy '(conf, 1, o, 1)))
      pure
        ( do
            ws <- w
            bs <- b
            wgs <- wg
            bgs <- bg
            pure (FFN ws bs wgs bgs)
        ),
    \ffn -> do
      p1 <- serializeMx (ffnW ffn)
      p2 <- serializeMx (ffnB ffn)
      p3 <- (snd (serializeMod (Proxy :: Proxy '(conf, i, o, 1)))) (ffnWGst ffn)
      p4 <- (snd (serializeMod (Proxy :: Proxy '(conf, 1, o, 1)))) (ffnBGst ffn)
      pure $ p1 >> p2 >> p3 >> p4
  )

-- | create a fully connected sgd network given a learning rate
-- m a wi hi di wo ho dpo
fullyConnectedSGD :: (KnownNat i, KnownNat o, Monad m, BlasM m a, RandomGen g, GradientDescentMethod m a conf mod i o 1, GradientDescentMethod m a conf mod2 1 o 1) => Proxy i -> Proxy o -> Layer m a (FFN a i o mod mod2) (FFNIn a i) (FFNGrad a i o) 1 i 1 1 o 1 conf (mod, mod2) g
fullyConnectedSGD nip nop = Layer forwardProp backwardProp avgGrad fsgdUpdate (fullyConnectedSGDInit nip nop) fsgdSerialize

fullyConnectedSGDInit :: (BlasM m mx, KnownNat i, KnownNat o, RandomGen g) => Proxy i -> Proxy o -> WeightInitializer g -> g -> m (FFN mx i o mod mod2, g)
fullyConnectedSGDInit nip nop f gen =
  do
    (nw', gen1) <- randomMx f gen nip nop (Proxy :: Proxy 1) (fromIntegral $ natVal nip) (fromIntegral $ natVal nop)
    --liftIO $ putStrLn $ "mxFromList fc sgd init 1 - " <> show (length nw)

    nb' <- konst 0.0
    --liftIO $ putStrLn "mxFromList fc sgd init e"
    pure (FFN nw' nb' Nothing Nothing, gen1)

-- o x i doubles
-- o doubles
-- o x i doubles
-- o doubles
-- i doubles
