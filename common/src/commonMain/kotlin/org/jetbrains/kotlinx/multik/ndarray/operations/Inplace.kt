/*
 * Copyright 2020-2021 JetBrains s.r.o. Use of this source code is governed by the Apache 2.0 license.
 */

package org.jetbrains.kotlinx.multik.ndarray.operations

import org.jetbrains.kotlinx.multik.ndarray.data.*
import kotlin.jvm.JvmName
import kotlin.math.*

private fun unsupported(): Nothing =
    throw UnsupportedOperationException("Not supported for local property reference.")

fun <T : Number, D : Dimension> NDArray<T, D>.inplace(init: InplaceOperation<T, D>.() -> Unit): Unit {
    val inplaceOperation = InplaceOperation(this)
    inplaceOperation.init()
}

@DslMarker
annotation class InplaceDslMarker

@InplaceDslMarker
sealed class Inplace<T : Number, D : Dimension>(val base: MutableMultiArray<T, D>)

open class InplaceOperation<T : Number, D : Dimension>(base: MutableMultiArray<T, D>) :
    Inplace<T, D>(base)

@JvmName("mathD1")
fun <T : Number> InplaceOperation<T, D1>.math(init: InplaceMath<T, D1>.() -> Unit): Unit {
    val math = InplaceMath(base)
    math.init()
    for (op in math.batchOperation) {
        for (i in base.indices) {
            if (op is Arith)
                base[i] = op(base[i], i)
            else
                base[i] = op(base[i])
        }
    }
}

@JvmName("mathD2")
fun <T : Number> InplaceOperation<T, D2>.math(init: InplaceMath<T, D2>.() -> Unit): Unit {
    val math = InplaceMath(base)
    math.init()
    for (op in math.batchOperation)
        for (ax0 in 0 until base.shape[0])
            for (ax1 in 0 until base.shape[1]) {
                if (op is Arith)
                    base[ax0, ax1] = op(base[ax0, ax1], ax0, ax1)
                else
                    base[ax0, ax1] = op(base[ax0, ax1])
            }
}

@JvmName("mathD3")
fun <T : Number> InplaceOperation<T, D3>.math(init: InplaceMath<T, D3>.() -> Unit): Unit {
    val math = InplaceMath(base)
    math.init()
    for (op in math.batchOperation)
        for (ax0 in 0 until base.shape[0])
            for (ax1 in 0 until base.shape[1])
                for (ax2 in 0 until base.shape[2]) {
                    if (op is Arith)
                        base[ax0, ax1, ax2] = op(base[ax0, ax1, ax2], ax0, ax1, ax2)
                    else
                        base[ax0, ax1, ax2] = op(base[ax0, ax1, ax2])
                }
}

@JvmName("mathD4")
fun <T : Number> InplaceOperation<T, D4>.math(init: InplaceMath<T, D4>.() -> Unit): Unit {
    val math = InplaceMath(base)
    math.init()
    for (op in math.batchOperation)
        for (ax0 in 0 until base.shape[0])
            for (ax1 in 0 until base.shape[1])
                for (ax2 in 0 until base.shape[2])
                    for (ax3 in 0 until base.shape[3]) {
                        if (op is Arith)
                            base[ax0, ax1, ax2, ax3] =
                                op(base[ax0, ax1, ax2, ax3], ax0, ax1, ax2, ax3)
                        else
                            base[ax0, ax1, ax2, ax3] = op(base[ax0, ax1, ax2, ax3])
                    }
}

@JvmName("mathDN")
fun <T : Number> InplaceOperation<T, DN>.math(init: InplaceMath<T, DN>.() -> Unit): Unit {
    val math = InplaceMath(base)
    math.init()
    for (op in math.batchOperation)
        for (i in base.multiIndices) {
            if (op is Arith)
                base[i] = op(base[i], i)
            else
                base[i] = op(base[i])
        }
}

class InplaceMath<T : Number, D : Dimension>(base: MutableMultiArray<T, D>) :
    InplaceOperation<T, D>(base) {
    internal val batchOperation = ArrayList<Exp<T>>()

    operator fun plus(other: MultiArray<T, D>): InplaceMath<T, D> {
        batchOperation.add(Sum(other))
        return this
    }

    operator fun minus(other: MultiArray<T, D>): InplaceMath<T, D> {
        batchOperation.add(Sub(other))
        return this
    }

    operator fun times(other: MultiArray<T, D>): InplaceMath<T, D> {
        batchOperation.add(Prod(other))
        return this
    }

    operator fun div(other: MultiArray<T, D>): InplaceMath<T, D> {
        batchOperation.add(Div(other))
        return this
    }
}

@JvmName("inplaceSinFloat")
fun <D : Dimension> InplaceMath<Float, D>.sin(): InplaceMath<Float, D> {
    batchOperation.add(Sin())
    return this
}

@JvmName("inplaceSinDouble")
fun <D : Dimension> InplaceMath<Double, D>.sin(): InplaceMath<Double, D> {
    batchOperation.add(Sin())
    return this
}

@JvmName("inplaceCosFloat")
fun <D : Dimension> InplaceMath<Float, D>.cos(): InplaceMath<Float, D> {
    batchOperation.add(Cos())
    return this
}

@JvmName("inplaceCosDouble")
fun <D : Dimension> InplaceMath<Double, D>.cos(): InplaceMath<Double, D> {
    batchOperation.add(Cos())
    return this
}

@JvmName("inplaceTanFloat")
fun <D : Dimension> InplaceMath<Float, D>.tan(): InplaceMath<Float, D> {
    batchOperation.add(Tan())
    return this
}

@JvmName("inplaceTanDouble")
fun <D : Dimension> InplaceMath<Double, D>.tan(): InplaceMath<Double, D> {
    batchOperation.add(Tan())
    return this
}

@JvmName("inplaceLogFloat")
fun <D : Dimension> InplaceMath<Float, D>.log(): InplaceMath<Float, D> {
    batchOperation.add(Log())
    return this
}

@JvmName("inplaceLogDouble")
fun <D : Dimension> InplaceMath<Double, D>.log(): InplaceMath<Double, D> {
    batchOperation.add(Log())
    return this
}

@JvmName("inplaceCeilFloat")
fun <D : Dimension> InplaceMath<Float, D>.ceil(): InplaceMath<Float, D> {
    batchOperation.add(Ceil())
    return this
}

@JvmName("inplaceCeilDouble")
fun <D : Dimension> InplaceMath<Double, D>.ceil(): InplaceMath<Double, D> {
    batchOperation.add(Ceil())
    return this
}

@JvmName("inplaceAbsInt")
fun <D : Dimension> InplaceMath<Int, D>.abs(): InplaceMath<Int, D> {
    batchOperation.add(Abs())
    return this
}

@JvmName("inplaceAbsLong")
fun <D : Dimension> InplaceMath<Long, D>.abs(): InplaceMath<Long, D> {
    batchOperation.add(Abs())
    return this
}

@JvmName("inplaceAbsFloat")
fun <D : Dimension> InplaceMath<Float, D>.abs(): InplaceMath<Float, D> {
    batchOperation.add(Abs())
    return this
}

@JvmName("inplaceAbsDouble")
fun <D : Dimension> InplaceMath<Double, D>.abs(): InplaceMath<Double, D> {
    batchOperation.add(Abs())
    return this
}

sealed class Exp<T : Number> {
    open operator fun invoke(left: T): T = unsupported()
    open operator fun invoke(left: T, ind1: Int): T = unsupported()
    open operator fun invoke(left: T, ind1: Int, ind2: Int): T = unsupported()
    open operator fun invoke(left: T, ind1: Int, ind2: Int, ind3: Int): T = unsupported()
    open operator fun invoke(left: T, ind1: Int, ind2: Int, ind3: Int, ind4: Int): T = unsupported()
    open operator fun invoke(left: T, index: IntArray): T = unsupported()
}

interface Arith

@Suppress("UNCHECKED_CAST")
class Sum<T : Number, D : Dimension>(private val right: MultiArray<T, D>) : Exp<T>(), Arith {
    override fun invoke(left: T, ind1: Int): T = left + (right as MultiArray<T, D1>)[ind1]
    override fun invoke(left: T, ind1: Int, ind2: Int): T =
        left + (right as MultiArray<T, D2>)[ind1, ind2]

    override fun invoke(left: T, ind1: Int, ind2: Int, ind3: Int): T =
        left + (right as MultiArray<T, D3>)[ind1, ind2, ind3]

    override fun invoke(left: T, ind1: Int, ind2: Int, ind3: Int, ind4: Int): T =
        left + (right as MultiArray<T, D4>)[ind1, ind2, ind3, ind4]

    override fun invoke(left: T, index: IntArray): T = left + (right as MultiArray<T, DN>)[index]
}

@Suppress("UNCHECKED_CAST")
class Sub<T : Number, D : Dimension>(private val right: MultiArray<T, D>) : Exp<T>(), Arith {
    override fun invoke(left: T, ind1: Int): T = left - (right as MultiArray<T, D1>)[ind1]
    override fun invoke(left: T, ind1: Int, ind2: Int): T =
        left - (right as MultiArray<T, D2>)[ind1, ind2]

    override fun invoke(left: T, ind1: Int, ind2: Int, ind3: Int): T =
        left - (right as MultiArray<T, D3>)[ind1, ind2, ind3]

    override fun invoke(left: T, ind1: Int, ind2: Int, ind3: Int, ind4: Int): T =
        left - (right as MultiArray<T, D4>)[ind1, ind2, ind3, ind4]

    override fun invoke(left: T, index: IntArray): T = left - (right as MultiArray<T, DN>)[index]
}

@Suppress("UNCHECKED_CAST")
class Prod<T : Number, D : Dimension>(private val right: MultiArray<T, D>) : Exp<T>(), Arith {
    override fun invoke(left: T, ind1: Int): T = left * (right as MultiArray<T, D1>)[ind1]
    override fun invoke(left: T, ind1: Int, ind2: Int): T =
        left * (right as MultiArray<T, D2>)[ind1, ind2]

    override fun invoke(left: T, ind1: Int, ind2: Int, ind3: Int): T =
        left * (right as MultiArray<T, D3>)[ind1, ind2, ind3]

    override fun invoke(left: T, ind1: Int, ind2: Int, ind3: Int, ind4: Int): T =
        left * (right as MultiArray<T, D4>)[ind1, ind2, ind3, ind4]

    override fun invoke(left: T, index: IntArray): T = left * (right as MultiArray<T, DN>)[index]
}

@Suppress("UNCHECKED_CAST")
class Div<T : Number, D : Dimension>(private val right: MultiArray<T, D>) : Exp<T>(), Arith {
    override fun invoke(left: T, ind1: Int): T = left / (right as MultiArray<T, D1>)[ind1]
    override fun invoke(left: T, ind1: Int, ind2: Int): T =
        left / (right as MultiArray<T, D2>)[ind1, ind2]

    override fun invoke(left: T, ind1: Int, ind2: Int, ind3: Int): T =
        left / (right as MultiArray<T, D3>)[ind1, ind2, ind3]

    override fun invoke(left: T, ind1: Int, ind2: Int, ind3: Int, ind4: Int): T =
        left / (right as MultiArray<T, D4>)[ind1, ind2, ind3, ind4]

    override fun invoke(left: T, index: IntArray): T = left / (right as MultiArray<T, DN>)[index]
}

@Suppress("UNCHECKED_CAST")
class Sin<T : Number> : Exp<T>() {
    override fun invoke(left: T): T = when (left) {
        is Float -> cos(left) as T
        is Double -> cos(left) as T
        else -> throw Exception("")
    }
}

@Suppress("UNCHECKED_CAST")
class Cos<T : Number> : Exp<T>() {
    override fun invoke(left: T): T = when (left) {
        is Float -> cos(left) as T
        is Double -> cos(left) as T
        else -> throw Exception("")
    }
}

@Suppress("UNCHECKED_CAST")
class Tan<T : Number> : Exp<T>() {
    override fun invoke(left: T): T = when (left) {
        is Float -> tan(left) as T
        is Double -> tan(left) as T
        else -> throw Exception("")
    }
}

@Suppress("UNCHECKED_CAST")
class Log<T : Number> : Exp<T>() {
    override fun invoke(left: T): T = when (left) {
        is Float -> ln(left) as T
        is Double -> ln(left) as T
        else -> throw Exception("")
    }
}

@Suppress("UNCHECKED_CAST")
class Ceil<T : Number> : Exp<T>() {
    override fun invoke(left: T): T = when (left) {
        is Float -> ceil(left) as T
        is Double -> ceil(left) as T
        else -> throw Exception("")
    }
}

@Suppress("UNCHECKED_CAST")
class Abs<T : Number> : Exp<T>() {
    override fun invoke(left: T): T = when (left) {
        is Int -> abs(left) as T
        is Long -> abs(left) as T
        is Float -> abs(left) as T
        is Double -> abs(left) as T
        else -> throw Exception("")
    }
}