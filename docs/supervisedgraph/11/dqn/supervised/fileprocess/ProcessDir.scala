package rl.dqn.supervised.fileprocess

import java.io.{File, FileOutputStream, ObjectOutputStream}

import akka.event.slf4j.Logger
import rl.dqn.supervised.fileprocess.withpeers.FileProcessorWithPeers
import rl.dqn.supervised.utils.ReplayDB

class ProcessDir(dirPath: String, objFileDirPath: String, fileNum: Int, numPerObj: Int) {
  private val logger = Logger("ProcessDir")
  
  var db = new ReplayDB()
  val dir = new File(dirPath)
  val fileName: String = dir.getName
  val objFileName: String = objFileDirPath + fileName
  val objFileNameType: String = ".obj"
  var fileSeq: Int = 0
  var totalNum: Int = 0

  def listFiles(): List[File] = new File(dirPath).listFiles.filter(_.isFile).toList

  def processFile(file: File): Unit = {
//    logger.info("-------------------------> Process " + file.getName)

    val processor = new FileProcessorWithPeers(file.getAbsolutePath)
    processor.readFile()
//    logger.info("========================> End of " + file.getName + " get objects " + processor.trans.length)

    processor.save(db)
  }

  def updateState(): Unit = {
    totalNum += db.datas.size()
    fileSeq += 1
    db = new ReplayDB()
  }

  def saveObj(): Unit = {
    val objStream = new ObjectOutputStream(new FileOutputStream(objFileName + "_" + fileSeq + objFileNameType))

    objStream.writeObject(db)
    objStream.close()

    updateState()
  }

  def processFiles(): Unit = {
    logger.info("====================> Get dir name " + fileName)
    logger.info("====================> Get obj file name " + objFileName)
    val files = listFiles()
    for (file <- files) {
      if (fileSeq < fileNum) {
        processFile(file)
//        saveObj()

        if (db.datas.size() >= numPerObj) {
          logger.info("++++++++++++++++++++++++++++++ TO save file " + db.datas.size())
          saveObj()
        }
      }
    }
    // give up remains

    logger.info("---------------------------------------Total: " + totalNum)
  }
}
