import System.Random
import Data.List
import Data.List.Utils
import qualified Data.Map as Map

data Cell = Empty | Wall | Can | ERob | CRob deriving (Show, Eq)

data Grid = Grid{   cells   :: [[Cell]]
                ,   loc     :: (Int, Int)} deriving (Show, Eq)

data Act = U | D | L | R | P | N

getCell :: Int -> Cell
getCell n = case n `mod` 10 of
            2 -> Can
            _ -> Empty

randomCell  :: IO Cell
randomCell = do n <- randomRIO(1,10)
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

--addRob :: [[Cell]] -> [[Cell]]
--addRob cs = (ERob : tail (head cs)): tail cs

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
move dir grd =  if checkMove dir (loc grd)
                then let    i = fst (loc grd)
                            j = snd (loc grd)
                            in case dir of
                                U -> let lc = (i, j-1) in Grid{ cells = addRob lc (removeRob (cells grd))
                                                                   , loc = lc}
                                D -> let lc = (i, j+1) in Grid{ cells = addRob lc (removeRob (cells grd))
                                                                   , loc = lc}
                                L -> let lc = (i-1, j) in Grid{ cells = addRob lc (removeRob (cells grd))
                                                                   , loc = lc}
                                R -> let lc = (i+1, j) in Grid{ cells = addRob lc (removeRob (cells grd))
                                                                   , loc = lc}
                                P -> let lc = loc grd in if isCan lc (cells grd)
                                                            then Grid{ cells = removeCan lc (cells grd)
                                                                     , loc = lc}
                                                            else grd
                                N -> grd


                else grd

checkMove :: Act -> (Int, Int) -> Bool
checkMove dir loc
    | fst loc <= 1 && snd loc <= 1 =    case dir of
                                            U -> False
                                            L -> False
                                            _ -> True
    | fst loc <= 1 =                    case dir of
                                            L -> False
                                            _ -> True
    | snd loc <= 1 =                    case dir of
                                            U -> False
                                            _ -> True
    | fst loc >= 9 && snd loc >= 9 =    case dir of
                                            D -> False
                                            R -> False
                                            _ -> True
    | fst loc >= 9 =                    case dir of
                                            R -> False
                                            _ -> True
    | snd loc >= 9 =                    case dir of
                                            D -> False
                                            _ -> True


main = do
    f <- randomField 8
    let ff = addWalls 10 f
        l = (1,1)
        saveme = Grid{cells = addRob (1,1) ff
                   ,loc = l }
        grid = Grid{cells = addRob (1,1) ff
                   ,loc = l }
        --table = Map.empty
    --printField (removeRob (addRob (4,6) (cells grid)))
    printField (cells grid)
    let t0 = move D grid
    printField (cells t0)
    let t1 = move R t0
    printField (cells t1)

