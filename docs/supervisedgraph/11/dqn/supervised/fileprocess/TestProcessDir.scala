package rl.dqn.supervised.fileprocess

import java.io._

import rl.dqn.reinforcement.dqn.config.Supervised._
import rl.dqn.supervised.utils.{DbTransition, ReplayDB}

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer

object TestProcessDir {

  def testLoad(): Unit = {
//    val objFileName = "/home/zf/workspaces/workspace_java/mjconv2/datasets/tenhoulogs/logs/db1/objs/validation/mjlog_pf4-20_n3_7.obj"
    val objFileName = "/home/zf/workspaces/workspace_java/mjconv2/datasets/tenhoulogs/logs/db3/validation/n10objsmjlog_pf4-20_n10_1.obj"

    val fileStream = new FileInputStream(objFileName)
    val objStream = new ObjectInputStream(fileStream)

    val db: ReplayDB = objStream.readObject().asInstanceOf[ReplayDB]

    println(db.datas.size())

    for(i <- 0 until 10) {
      println(db.datas.get(i).getObservation()(0))
    }
  }

  def testDir(): Unit = {
    val dirPath = "/home/zf/workspaces/workspace_java/mjconv2/datasets/tenhoulogs/logs/db1/xmls/mjlog_pf4-20_n9"
    val objDirPath = "/home/zf/workspaces/workspace_java/mjconv2/datasets/tenhoulogs/logs/db1/objs/n9objs/"

    val processor = new ProcessDir(dirPath, objDirPath, 10, 1024 * 8)
    processor.processFiles()
  }

  def testDirWoAccept(index: Int): Unit = {
    val dirPath = "/home/zf/workspaces/workspace_java/mjconv2/datasets/tenhoulogs/logs/xmlfiles/train/"
    val objDirPath = "/home/zf/workspaces/workspace_java/mjconv2/datasets/tenhoulogs/logs/db2/train/n" + index + "objs"

    val processor = new ProcessDirWoAction(dirPath, objDirPath, 10, 1024 * 8)
    processor.processFiles()
  }

  def testDirWoAcceptPlayer(index: Int): Unit = {
    val dirPath = "/home/ec2-user/tenhoulogs/xmlfiles/train/"
    val objDirPath = "/home/ec2-user/tenhoulogs/logs/db4/dqrn/player/train/nall" + "objs"

    val processor = new ProcessDirWoActionWhPlayer(dirPath, objDirPath, Int.MaxValue, 1024 * 8)
    processor.processFiles()
  }


  def testDirWoAcceptAll(): Unit = {
    val dirPath = "/home/zf/workspaces/workspace_java/tenhoulogs/logs/xmlfiles/"
    val objDirPath = "/home/zf/workspaces/workspace_java/tenhoulogs/logs/db4/raws/test"
    val dirs = new File(dirPath).listFiles().filter(_.isDirectory).toList

    dirs.foreach(dir => {
      println("==============================================================> Process " + dir.getAbsolutePath)
      new ProcessDirWoAction(dir.getAbsolutePath, objDirPath, Int.MaxValue, 1024  * 8).processFiles()
    })
  }

  def testCount(): Unit = {
    val dirPath = "/home/zf/workspaces/workspace_java/mjconv2/datasets/tenhoulogs/logs/db2/train/"
    val counts = Array.fill[Long](ActionLenWoAccept)(0)

    val files = new File(dirPath).listFiles().filter(_.isFile).toList

    files.foreach(file => {
      val objStream = new ObjectInputStream(new FileInputStream(file))
      val db = objStream.readObject().asInstanceOf[ReplayDB]

      for(i <- 0 until db.datas.size()) {
        counts(db.datas.get(i).getAction) += 1
      }
    })

    counts.foreach(println)
  }

  def saveObjs(trans: Array[mutable.LinkedList[DbTransition]], path: String, index: Int, capacity: Int): Unit = {
    val db = new ReplayDB()
    for (_ <- 0 until capacity) {
      for (j <- trans.indices) {
        if (trans(j).nonEmpty) {
          db.add(trans(j).head)
          trans(j) = trans(j).tail
        }
      }
    }

    val fileName = path + index + ".obj"
    val output = new ObjectOutputStream(new FileOutputStream(fileName))
    output.writeObject(db)
  }

  def testBalance(): Unit = {
    val dirPath = "/home/zf/workspaces/workspace_java/tenhoulogs/logs/db4/raws/"
    val destDirPath = "/home/zf/workspaces/workspace_java/tenhoulogs/logs/db4/train/"
    var fileCount: Int = 0
    val countPerFile = 1000
    val fileCapacity = 80
//    val totalNum = 10000
    val trans = new Array[mutable.LinkedList[DbTransition]](ActionLenWoAccept)
    for (i <- 0 until ActionLenWoAccept) {
      trans(i) = mutable.LinkedList.empty[DbTransition]
    }

    val files = new File(dirPath).listFiles().filter(_.isFile).toList
    files.foreach(file => {
      if (fileCount < fileCapacity) {
        val db = new ObjectInputStream(new FileInputStream(file)).readObject().asInstanceOf[ReplayDB]
        for (i <- 0 until db.datas.size()) {
          trans(db.datas.get(i).getAction) = trans(db.datas.get(i).getAction) :+ db.datas.get(i)
        }

        if (trans.count(tran => {
          tran.length > countPerFile
        }) >= (ActionLenWoAccept - 4)) {
          saveObjs(trans, destDirPath, fileCount, countPerFile)
          fileCount += 1
        }
      }
    })

    saveObjs(trans, destDirPath, fileCount, countPerFile)
  }

  def testArrayBuffer(): Unit = {
    val tester = ArrayBuffer[Int]()
    tester += 1
    tester += 2
    tester += -3
    tester += 10
    tester += -4

    println(tester)
    // Keep sequence
  }

  def main(args: Array[String]): Unit = {
//    testDir()
//    testLoad()
//    testDirWoAccept(10)
//    testCount()
//    testBalance()
//    testDirWoAcceptAll()
//    testArrayBuffer()
    val fileSeqs = Array[Int](1, 2, 4, 5, 6, 7, 9, 10, 12)
    fileSeqs.foreach(seq => testDirWoAcceptPlayer(seq))

//    testDirWoAcceptPlayer(1)
  }
}
