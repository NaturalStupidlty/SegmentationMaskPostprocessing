import CoreImage
import Foundation
import Accelerate

func computeCentroid(mask: CVPixelBuffer, depthMap: CVPixelBuffer, sidewalkLabel: Int = 1, useDBSCAN: Bool = false) -> (CGFloat, CGFloat)? {
    guard CVPixelBufferGetWidth(mask) == CVPixelBufferGetWidth(depthMap),
          CVPixelBufferGetHeight(mask) == CVPixelBufferGetHeight(depthMap) else {
        return nil
    }

    CVPixelBufferLockBaseAddress(mask, .readOnly)
    CVPixelBufferLockBaseAddress(depthMap, .readOnly)

    defer {
        CVPixelBufferUnlockBaseAddress(mask, .readOnly)
        CVPixelBufferUnlockBaseAddress(depthMap, .readOnly)
    }

    let width = CVPixelBufferGetWidth(mask)
    let height = CVPixelBufferGetHeight(mask)
    
    guard let maskBaseAddress = CVPixelBufferGetBaseAddress(mask)?.assumingMemoryBound(to: UInt8.self),
          let depthBaseAddress = CVPixelBufferGetBaseAddress(depthMap)?.assumingMemoryBound(to: Float32.self) else {
        return nil
    }

    var xValues: [CGFloat] = []
    var yValues: [CGFloat] = []
    var zValues: [Float32] = []

    for y in 0..<height {
        for x in 0..<width {
            let pixelIndex = y * width + x
            let label = Int(maskBaseAddress[pixelIndex])
            if label == sidewalkLabel {
                xValues.append(CGFloat(x))
                yValues.append(CGFloat(y))
                zValues.append(depthBaseAddress[pixelIndex])
            }
        }
    }

    if useDBSCAN {
        let clusteredPoints = zip(xValues, zip(yValues, zValues)).map { ($0.0, $0.1.0, $0.1.1) }
        guard let largestCluster = performDBSCAN(points: clusteredPoints) else { return nil }
        let (xCentroid, yCentroid, _) = computeMeanCluster(cluster: largestCluster)
        return (xCentroid, yCentroid)
    } else {
        let xCentroid = median(of: xValues)
        let yCentroid = median(of: yValues)
        return (xCentroid, yCentroid)
    }
}

func median(of values: [CGFloat]) -> CGFloat {
    let sortedValues = values.sorted()
    let count = sortedValues.count
    if count % 2 == 0 {
        return (sortedValues[count / 2 - 1] + sortedValues[count / 2]) / 2
    } else {
        return sortedValues[count / 2]
    }
}

func performDBSCAN(points: [(CGFloat, CGFloat, Float32)]) -> [(CGFloat, CGFloat, Float32)]? {
    // todo: dbscan
    return points 
}

func computeMeanCluster(cluster: [(CGFloat, CGFloat, Float32)]) -> (CGFloat, CGFloat, Float32) {
    let sumX = cluster.reduce(0) { $0 + $1.0 }
    let sumY = cluster.reduce(0) { $0 + $1.1 }
    let sumZ = cluster.reduce(0) { $0 + $1.2 }
    let count = CGFloat(cluster.count)
    return (sumX / count, sumY / count, sumZ / Float32(count))
}