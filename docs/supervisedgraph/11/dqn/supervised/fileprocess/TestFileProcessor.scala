package rl.dqn.supervised.fileprocess
import akka.event.slf4j.Logger
import rl.dqn.reinforcement.dqn.config.Supervised._

object TestFileProcessor {

  private[this] val dir = "/home/zf/workspaces/workspace_java/mjconv2/datasets/tenhoulogs/xmls/"
  //TODO: Why can not be defined as Array[Any]?
  def printArray(a: Array[Int]): Unit = {
    a.foreach(i => print(i + ", "))
    println
  }

  def createTester(fileName: String): FileProcessor = {
    val tester = new FileProcessor(fileName)
    tester.readFile()

    tester
  }

  def testInit(): Unit = {
    val fileName = dir + "testinit.xml"
    val tester = new FileProcessor(fileName)
    tester.readFile()

    println("dora ")
    println(tester.doraValue(28) == DoraValue)
    println(tester.doraValue(27) == 0)

    println("oya")
    println(tester.oya == 1)

    println("States")
    val expectedStates = new Array[Array[Int]](PlayerNum)
    for(i <- expectedStates.indices) {
      expectedStates(i) = new Array[Int](StateLen)
    }

    val index0 = Array[Int](27, 31, 6, 7, 17, 20, 15, 3, 28, 21, 4, 9, 17)
    index0.foreach(i => expectedStates(0)(i) += 1)
    expectedStates(0)(28) += 10
    println(tester.currentStats(0).deep == expectedStates(0).deep)

    val index1 = Array[Int](12, 29, 13, 30, 8, 9, 4, 3, 15, 5, 6, 18, 20)
    index1.foreach(i => expectedStates(1)(i) += 1)
    expectedStates(1)(4) += 10
    expectedStates(1)(OyaIndex) = 1
    println(tester.currentStats(1).deep == expectedStates(1).deep)


    val index2 = Array[Int](9, 24, 6, 27, 4, 0, 26, 32, 32, 5, 11, 33, 7)
    index2.foreach(i => expectedStates(2)(i) += 1)
    println(tester.currentStats(2).deep == expectedStates(2).deep)


    val index3 = Array[Int](25, 25, 1, 29, 22, 27, 29, 29, 10, 2, 7, 30, 14)
    index3.foreach(i => expectedStates(3)(i) += 1)
    println(tester.currentStats(3).deep == expectedStates(3).deep)

    println("ten")
    val expectedTen = Array[Int](254, 288, 229, 229)
    println(expectedTen.deep == tester.tens.deep)
  }

  def testAgari(): Unit = {
    val tester = createTester(dir + "testagari.xml")
    val trans = tester.trans
    val expectedRewards = Array[Double](0, -32, 52, 0)

    for (i <- trans.indices) {
      println(expectedRewards(i) == trans(i).getReward)
    }

    println("Who")
    for (i <- trans.indices) {
      if (i == 2) {
        println(trans(i).getAction == Ron)
      }else {
        println(trans(i).getAction == NOOP)
      }
    }
  }

  def testR(): Unit = {
    val tester = createTester(dir + "testryuukyoku.xml")
    val trans = tester.trans
    val expectedRewards = Array[Double](15, 15, -15, -15)

    for (i <- trans.indices) {
      println(expectedRewards(i) == trans(i).getReward)
    }

    trans.foreach(tran => {
      println(tran.getAction == NOOP)
    })
  }

  def testReach(): Unit = {
    val tester = createTester(dir + "testreachstep1.xml")
    val trans = tester.trans

    println("Reward")
    println(trans.head.getReward == ReachReward)

    println("Reach flag")
    tester.isReach.foreach(x => println(!x))

    println("Read flag")
    val tester2 = createTester(dir + "testreach.xml")
    for (i <- tester2.isReach.indices) {
      if (i == 2) {
        println(tester2.isReach(i))
      }else {
        println(!tester2.isReach(i))
      }
    }
  }

  def testPlayer(): Unit = {
    val tester = createTester(dir + "testplayer.xml")
    val winnerTrans = tester.trans(tester.trans.length - 2)
    val state = winnerTrans.getNextObservation

    // dora = 0
    //  hai="12,17,18,22,23,27,44,51,54,57,60,66,96,97"
    val expectedTile = Array[Int](3, 4, 4, 5, 5, 6, 11, 12, 13, 14, 15, 16, 24, 24)
    val expectedState = Array.fill[Double](StateLen)(0)
    expectedTile.foreach(t => {
      expectedState(t) += 1
    })
    expectedState(ReachIndex) = 1

    for (i <- expectedState.indices) {
      println(expectedState(i) + " == " + state.getDouble(i) + "? " + (expectedState(i) == state.getDouble(i)))
    }
  }

  def testPlayerStep(): Unit = {
    val tester = createTester(dir + "testplayerstep.xml")
    val expectedStates = new Array[Array[Int]](PlayerNum)
    for(i <- expectedStates.indices) {
      expectedStates(i) = Array.fill[Int](StateLen)(0)
    }
    val expectedTile = new Array[Array[Int]](PlayerNum)
    expectedTile(0) = Array[Int](7,0,10,22,20,2,13,2,18,6,18,19,16)
    expectedTile(1) = Array[Int](32,19,3,20,25,29,19,1,9,7,20,11,23)
    expectedTile(2) = Array[Int](27,24,20,15,5,12,13,4,25,5,29,24,14)
    expectedTile(3) = Array[Int](17,31,14,5,22,15,19,17,10,21,17,9,6)

    // oya = 3, dora = 1
    for (i <- expectedTile.indices) {
      expectedTile(i).foreach(tile => {
        expectedStates(i)(tile) += 1
      })
    }
    expectedStates(3)(OyaIndex) = 1
    expectedStates(1)(1) += expectedStates(1)(1) * DoraValue

    val states = tester.currentStats
    for (i <- states.indices) {
      println(states(i).deep == expectedStates(i).deep)
    }

    printArray(states(1))
    printArray(expectedStates(1))
  }

  def testDora(): Unit = {
    val tester = createTester(dir + "testdora.xml")
    val trans = tester.trans

    val expectedStates = new Array[Array[Int]](PlayerNum)
    for(i <- expectedStates.indices) {
      expectedStates(i) = new Array[Int](StateLen)
    }

    // dora = 109, 13, 16, 52, 88
    val index0 = Array[Int](27, 31, 6, 7, 17, 20, 15, 3, 28, 21, 4, 9, 17)
    index0.foreach(i => expectedStates(0)(i) += 1)
    expectedStates(0)(28) += DoraValue
    expectedStates(0)(4) += DoraValue
    printArray(expectedStates(0))
    printArray(tester.currentStats(0))
    println(tester.currentStats(0).deep == expectedStates(0).deep)

    val index1 = Array[Int](12, 29, 13, 30, 8, 9, 4, 3, 15, 5, 6, 18, 20)
    index1.foreach(i => expectedStates(1)(i) += 1)
    expectedStates(1)(4) += DoraValue * 2
    expectedStates(1)(OyaIndex) = 1
    printArray(expectedStates(1))
    printArray(tester.currentStats(1))
    println(tester.currentStats(1).deep == expectedStates(1).deep)


    val index2 = Array[Int](9, 24, 6, 27, 4, 0, 26, 32, 32, 5, 11, 33, 7)
    index2.foreach(i => expectedStates(2)(i) += 1)
    expectedStates(2)(4) += DoraValue
    printArray(expectedStates(2))
    printArray(tester.currentStats(2))
    println(tester.currentStats(2).deep == expectedStates(2).deep)


    val index3 = Array[Int](25, 25, 1, 29, 22, 27, 29, 29, 10, 2, 7, 30, 14)
    index3.foreach(i => expectedStates(3)(i) += 1)
    printArray(expectedStates(3))
    printArray(tester.currentStats(3))
    println(tester.currentStats(3).deep == expectedStates(3).deep)


    val tester2 = createTester(dir + "testdoradrop.xml")
    expectedStates(1)(4) = 0
    for (i <- expectedStates.indices) {
      println(tester2.currentStats(i).deep == expectedStates(i).deep)
    }
  }

  def getDefaultState(expectedTile: Array[Int]): Array[Int] = {
    val expectedState = Array.fill[Int](StateLen)(0)
    expectedTile.foreach(tile => expectedState(tile) += 1)

    expectedState
  }

  def getDefaultState(expectedTile: Array[Array[Int]]): Array[Array[Int]] = {
    val expectedStates = new Array[Array[Int]](PlayerNum)
//    for(i <- expectedStates.indices) {
//      expectedStates(i) = Array.fill[Int](StateLen)(0)
//    }
//    for (i <- expectedTile.indices) {
//      expectedTile(i).foreach(tile => {
//        expectedStates(i)(tile) += 1
//      })
//    }
    for(i <- expectedTile.indices) {
      expectedStates(i) = getDefaultState(expectedTile(i))
    }

    expectedStates
  }


  def testPong(): Unit = {
    val tester = createTester(dir + "testpong.xml")

    //dora = 20, 16, 88, 52 original
//    expectedTile(0) = Array[Int](20, 23, 23, 9, 9, 30, 9, 27, 20, 6, 24, 11, 11)
//    expectedTile(1) = Array[Int](27, 28, 32, 26, 24, 33, 15, 2, 22, 17, 11, 8, 14)
//    expectedTile(2) = Array[Int](22, 14, 7, 27, 4, 27, 13, 2, 0, 20, 0, 13, 6)
//    expectedTile(3) = Array[Int](16, 19, 4, 26, 32, 1, 32, 6, 31, 4, 29, 7, 16)
    val expectedTile = new Array[Array[Int]](PlayerNum)

    expectedTile(0) = Array[Int](20, 23, 23, 9, 9, 25, 9, 12, 20, 6, 24, 11, 11)
    expectedTile(1) = Array[Int](28, 28, 32, 26, 24, 33, 15, 2, 22, 17, 11, 11, 14)
    expectedTile(2) = Array[Int](22, 14, 7, 27, 4, 27, 15, 2, 0, 20, 0, 13, 6, 27)
    expectedTile(3) = Array[Int](16, 19, 4, 10, 32, 1, 32, 6, 31, 4, 29, 7, 16)

    val expectedStates = getDefaultState(expectedTile)
    expectedStates(0)(OyaIndex) = 1
    expectedStates(0)(20) += 2 * DoraValue
    expectedStates(1)(22) += DoraValue
    expectedStates(2)(13) += DoraValue
    expectedStates(2)(4) += DoraValue
    expectedStates(2)(20) += DoraValue
    expectedStates(2)(27) = 3 * MValue

    for (i <- expectedStates.indices) {
      println(tester.currentStats(i).deep == expectedStates(i).deep)
      printArray(tester.currentStats(i))
      printArray(expectedStates(i))
    }

    val trans = tester.trans
    val tran = trans(trans.length - 1)
    println(tran.getReward == DefaultReward)
    println(tran.getAction == Pong)
    println("Next")
    for (i <- expectedStates(2).indices) {
      if(expectedStates(2)(i) != tran.getNextObservation.getDouble(i)) {
        print(false)
      }
    }
    println("")
    println("Observation")
    expectedStates(2)(27) -= 1
    for (i <- expectedStates(2).indices) {
      if (expectedStates(2)(i) != tran.getObservation.head.getDouble(i)) {
        print(false)
      }
    }
    println("")
  }

  /*
  41547: pong 108
44139: pong 114
16939: pong 44
20799: chow 34
34409: pong 89
14839: chow 27
   */
  //TODO: (aka)dora tiles can not be used in chow
  def testChow(): Unit = {
    val tester = createTester(dir + "testchow.xml")

    val expectedTile = new Array[Array[Int]](PlayerNum)

// initial value
//    expectedTile(0) = Array[Int](11, 4, 14, 17, 22, 25, 4, 3, 15, 6, 19, 30, 3)
//    expectedTile(1) = Array[Int](0, 12, 8, 18, 22, 1, 9, 9, 18, 17, 33, 24, 21)
//    expectedTile(2) = Array[Int](33, 11, 10, 5, 22, 4, 10, 27, 20, 32, 16, 12, 21)
//    expectedTile(3) = Array[Int](18, 14, 17, 20, 13, 29, 30, 29, 29, 27, 26, 33, 25)

    expectedTile(0) = Array[Int](11, 4, 14, 5, 22, 23, 4, 3, 15, 6, 13, 30, 3)
    expectedTile(1) = Array[Int](0, 8, 8, 18, 22, 1, 9, 9, 18, 17, 16, 24, 23)
    expectedTile(2) = Array[Int](16, 11, 10, 5, 22, 4, 10, 28, 20, 10, 16, 12, 21, 6)
    expectedTile(3) = Array[Int](18, 14, 20, 20, 13, 29, 19, 29, 29, 21, 26, 26, 25)

    //oya = 1, 24, 16, 88, 52
    val expectedStates = getDefaultState(expectedTile)
    expectedStates(0)(4) += DoraValue
    expectedStates(0)(22) += DoraValue
    expectedStates(1)(24) += DoraValue
    expectedStates(1)(OyaIndex) = 1
//    expectedStates(2)(20) += DoraValue
    expectedStates(2)(4) += MValue - 1
    expectedStates(2)(5) += MValue - 1
    expectedStates(2)(6) += MValue - 1
    expectedStates(3)(13) += DoraValue

    for (i <- expectedStates.indices) {
      println(tester.currentStats(i).deep == expectedStates(i).deep)
      printArray(tester.currentStats(i))
      printArray(expectedStates(i))
    }


    val trans = tester.trans
    val tran = trans(trans.length - 1)
    println(tran.getReward == DefaultReward)
    println(tran.getAction == Chow)
    println("Next")
    for (i <- expectedStates(2).indices) {
      if(expectedStates(2)(i) != tran.getNextObservation.getDouble(i)) {
        print(false)
      }
    }
    println("")
    println("Observation")
    expectedStates(2)(4) -= MValue - 1
    expectedStates(2)(5) -= MValue - 1
    expectedStates(2)(6) = 0
    for (i <- expectedStates(2).indices) {
      if (expectedStates(2)(i) != tran.getObservation.head.getDouble(i)) {
        print(false)
      }
    }
    println("")
  }

  def testScene1(): Unit = {
    val tester = createTester(dir + "testscene1.xml")

    //dora = 24, 16, 52, 88
    //oya = 1
    val expectedTile = Array[Int](10, 10, 10, 11, 12, 13, 16, 16, 20, 21, 22, 4, 5, 6)
    val expectedStates = getDefaultState(expectedTile)
    expectedStates(13) += DoraValue
    expectedStates(4) = MValue
    expectedStates(5) = MValue
    expectedStates(6) = MValue

    val trans = tester.trans
    val tran = trans(trans.length - 2)
    println(tran.getReward == 20)
    println(tran.getAction == Ron)
    println("Next")
    for (i <- expectedStates.indices) {
        println(expectedStates(i) + " == " + tran.getNextObservation.getDouble(i) + "? " + (expectedStates(i) == tran.getNextObservation.getDouble(i)))
    }
    println("")
  }

  def testScene(): Unit = {
    val tester = createTester(dir + "testscene.xml")
    //dora = 28, 16, 52, 88
    //oya = 1
    val expectedTile = Array[Int](3, 4, 5, 6, 7, 8, 9, 9, 11, 11, 11, 32, 32, 32)
    val expectedStates = getDefaultState(expectedTile)
    expectedStates(ReachIndex) = 2
//    expectedStates(13) += DoraValue
//    expectedStates(4) = MValue
//    expectedStates(5) = MValue
//    expectedStates(6) = MValue

    val trans = tester.trans
    val tran = trans(trans.length - 2)
    println(tran.getReward == 52)
    println(tran.getAction == Ron)
    println("Next")
    for (i <- expectedStates.indices) {
      if (expectedStates(i) != tran.getNextObservation.getDouble(i)) {
        print(false)
      }
    }
    println("")
  }

  def testFile(): Unit = {
    // Just to see if any exception
    // test.xml and test2.xml are good test objects
    createTester(dir + "test2.xml")
  }

  def main(args: Array[String]): Unit = {
    System.out.println(NTagProcessor.VisitN(2, 28673));
//    testFile()
  }
}
