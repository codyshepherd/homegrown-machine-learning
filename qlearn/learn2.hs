import System.Random
import Data.List
import Data.List.Utils
import Data.Map (Map)
import qualified Data.Map as Map

n = 5000
m = 200
eta = 0.2
gamma = 0.9

data Cell = Empty | Wall | Can | ERob | CRob deriving (Show, Eq)

data Grid = Grid{   cells   :: [[Cell]]
                ,   loc     :: (Int, Int)
                ,   reward  :: Int
                ,   episode :: Int
                ,   step    :: Int
                ,   epsilon :: Float
                ,   qtable  :: Map String [Int]} deriving (Show, Eq)

data Act = U | D | L | R | P deriving (Show, Eq)

data State = State{ north :: Cell
                  , south :: Cell
                  , east  :: Cell
                  , west  :: Cell
                  , here  :: Cell
                  , key   :: String} deriving (Show, Eq)

getCell :: Int -> Cell
getCell n = case n `mod` 2 of
            0 -> Can
            _ -> Empty

getAction :: Int -> Act
getAction n =  case n `mod` 5 of
                0 -> U
                1 -> D
                2 -> L
                3 -> R
                4 -> P

{-
getBool :: Int -> Bool
getBool n = case n `mod` 2 of
                0 -> True
                1 -> False
-}

randomCell  :: IO Cell
randomCell = do n <- randomRIO(1,2)
                return (getCell n)

infixr `times`
times       :: Int -> IO a -> IO [a]
n `times` action = sequence (replicate n action)

randomField :: Int -> IO [[Cell]]
randomField d = d `times` d `times` randomCell

horizWall :: Int -> [Cell]
horizWall n = [Wall | i<-[1..n]]

addWalls :: Int -> [[Cell]] -> [[Cell]]
addWalls n cs = let l =  horizWall n : [[Wall] ++ c ++ [Wall] | c <- cs] in l ++ [horizWall n]

lastN' :: Int -> [a] -> [a]
lastN' n xs = foldl' (const . drop 1) xs (drop n xs)

randomLoc :: Int -> IO (Int, Int)
randomLoc n = do i <- randomRIO(1,n)
                 j <- randomRIO(1,n)
                 return (i,j)

randomAct :: IO Act
randomAct = do n <- randomRIO(1,5)
               return (getAction n)

randomBool :: Int -> IO Bool
randomBool p = do n <- randomRIO(1,100)
                  return (if n > p then True else False)

randomActions :: Int -> Int -> IO [Act]
randomActions n m = (n*m) `times` randomAct

isActionRandom :: Float -> Int -> Int -> Int -> IO [Bool]
isActionRandom eps n m step = mapM randomBool [floor (eps-(0.01* fromIntegral y) * 100.0)
                                | x <- [1..(n*m)]
                                , y <- [x `div` (m*step)]]

addRob :: (Int, Int) -> [[Cell]] -> [[Cell]]
addRob loc cs = let r = if cs !! (fst loc) !! (snd loc) == Empty then ERob else CRob
                    i = fst loc
                    j = snd loc
                    in take j cs ++ [take i (cs !! j) ++ [r] ++ lastN' (9-i) (cs !! j)] ++ lastN' (9-j) cs

removeRob :: [[Cell]] -> [[Cell]]
removeRob cs = let gs = [replace [ERob] [Empty] c | c <- cs] in [replace [CRob] [Can] g | g <- gs]

removeCan :: (Int, Int) -> [[Cell]] -> [[Cell]]
removeCan loc cs = let r = if cs !! (fst loc) !! (snd loc) == CRob then ERob else Empty
                       i = fst loc
                       j = snd loc
                    in take j cs ++ [take i (cs !! j) ++ [r] ++ lastN' (9-i) (cs !! j)] ++ lastN' (9-j) cs

isCan :: (Int, Int) -> [[Cell]] -> Bool
isCan loc cs = let  i = fst loc
                    j = snd loc
                    in if cs !! j !! i == Can || cs !! j !! i == CRob then True else False

letterOf :: Cell -> String
letterOf Empty = "[ ]"
letterOf Wall = " = "
letterOf Can = "[c]"
letterOf ERob = "[o]"
letterOf CRob = "[8]"

listValues :: [[Cell]] -> [[String]]
listValues xs = map (map letterOf) xs

printField :: [[Cell]] -> IO ()
printField xs = mapM_ putStrLn [ intercalate "" a | a<-listValues xs]

getKey :: [Cell] -> String
getKey [] = ""
getKey (c:cs) =
    case c of
        Empty -> '0':getKey cs
        Wall -> '1':getKey cs
        Can -> '2':getKey cs
        ERob -> '3':getKey cs
        CRob -> '4':getKey cs

move :: Act -> Grid -> Grid
move dir grd =   let    rw = checkMove dir grd
                        qt = qtable grd
                        i = fst (loc grd)
                        j = snd (loc grd)
                        epi = episode grd
                        st = step grd
                        eps = epsilon grd
                        in case dir of
                            U -> let lc = (i, j-1) in Grid{ cells = addRob lc (removeRob (cells grd))
                                                          , loc=lc, reward=rw, episode=epi, step=st+1,epsilon=eps, qtable=qt}
                            D -> let lc = (i, j+1) in Grid{ cells = addRob lc (removeRob (cells grd))
                                                          , loc=lc, reward=rw, episode=epi, step=st+1,epsilon=eps, qtable=qt}
                            L -> let lc = (i-1, j) in Grid{ cells = addRob lc (removeRob (cells grd))
                                                          , loc=lc, reward=rw, episode=epi, step=st+1,epsilon=eps, qtable=qt}
                            R -> let lc = (i+1, j) in Grid{ cells = addRob lc (removeRob (cells grd))
                                                          , loc=lc, reward=rw, episode=epi, step=st+1,epsilon=eps, qtable=qt}
                            P -> let lc = loc grd in if isCan lc (cells grd)
                                                        then Grid{ cells = removeCan lc (cells grd)
                                                                 , loc=lc, reward=rw, episode=epi, step=st+1,epsilon=eps, qtable=qt}
                                                        else Grid{ cells = cells grd
                                                                 , loc = lc
                                                                 , reward = rw
                                                                 , episode = epi
                                                                 , step = st+1
                                                                 , epsilon = eps
                                                                 , qtable=qt }

checkMove :: Act -> Grid -> Int
checkMove dir grd
    | fst (loc grd) <= 1 && snd (loc grd) <= 1 =    case dir of
                                                U -> -5
                                                L -> -5
                                                _ -> 0
    | fst (loc grd) <= 1 =                    case dir of
                                                L -> -5
                                                _ -> 0
    | snd (loc grd) <= 1 =                    case dir of
                                                U -> -5
                                                _ -> 0
    | fst (loc grd) >= 9 && snd (loc grd) >= 9 =    case dir of
                                                D -> -5
                                                R -> -5
                                                _ -> 0
    | fst (loc grd) >= 9 =                    case dir of
                                                R -> -5
                                                _ -> 0
    | snd (loc grd) >= 9 =                    case dir of
                                                D -> -5
                                                _ -> 0
    | otherwise =                                   case cells grd !! snd (loc grd) !! fst (loc grd) of
                                                CRob -> if dir == P then 10 else 0
                                                ERob -> if dir == P then -1 else 0
                                                _ -> 0

getState :: Grid -> State
getState grd = let  i = fst (loc grd)
                    j = snd (loc grd)
                    in let  n = cells grd !! (j-1)  !! i
                            s = cells grd !! (j+1)  !! i
                            e = cells grd !! j      !! (i+1)
                            w = cells grd !! j      !! (i-1)
                            h = cells grd !! j      !! i
                            k = getKey [n,s,e,w,h]
                            in State{ north = n
                                    , south = s
                                    , east = e
                                    , west = w
                                    , here = h
                                    , key = k}

newQTable :: Map String [Int]
newQTable = let strings = [ [n] ++ [s] ++ [e] ++ [w] ++ [h] | n <- ['0'..'2'], s <- ['0'..'2']
                                                  , e <- ['0'..'2'], w <- ['0'..'2'], h <- ['0'..'4']]
                in Map.fromList (map makePair strings)
                    where makePair x = (x, [0,0,0,0,0])

getNewAction :: Grid -> Act
getNewAction grd = U


--learn :: State -> State -> Act -> Grid -> Grid
--learn s sprime a grd = []

{-
execEpisode :: Int -> Grid -> Grid
execEpisode n grd = do s <- getState grd
                       act <- getNewAction grd
                       grdprime <- move act grd
                       sprime <- getState grdprime
                       grdprime
                       --learn s sprime act grdprime
-}


main = do
    f <- randomField 8
    l <- randomLoc 8
    randoms <- randomActions n m
    isRandoms <- isActionRandom 1 n m 50
    let
        ff = addWalls 10 f
        --saveme = Grid{cells = addRob l ff
        --           ,loc = l }
        table = newQTable
        grid = Grid{cells = addRob l ff
                   ,loc = l
                   ,reward = 0
                   ,episode = 0
                   ,step = 0
                   ,epsilon = 1
                   ,qtable = table}
        st = getState grid
        numEpisodes = 10
    print (take 10 randoms)
    print (take 10 isRandoms)
    --printField (removeRob (addRob (4,6) (cells grid)))
    {-
    printField (cells grid)
    let t0 = move D grid
    printField (cells t0)
    let t1 = move R t0
    printField (cells t1)
     -}