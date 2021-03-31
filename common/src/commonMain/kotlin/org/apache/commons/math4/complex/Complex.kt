/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.apache.commons.math4.complex

import org.apache.commons.math4.FieldElement
import org.apache.commons.math4.util.Precision
import kotlin.math.*

/**
 * Representation of a Complex number, i.e. a number which has both a
 * real and imaginary part.
 * <br></br>
 * Implementations of arithmetic operations handle `NaN` and
 * infinite values according to the rules for [java.lang.Double], i.e.
 * [.equals] is an equivalence relation for all instances that have
 * a `NaN` in either real or imaginary part, e.g. the following are
 * considered equal:
 *
 *  * `1 + NaNi`
 *  * `NaN + i`
 *  * `NaN + NaNi`
 *
 * Note that this is in contradiction with the IEEE-754 standard for floating
 * point numbers (according to which the test `x == x` must fail if
 * `x` is `NaN`). The method
 * [ equals for primitive double][org.apache.commons.math4.util.Precision.equals] in [org.apache.commons.math4.util.Precision]
 * conforms with IEEE-754 while this class conforms with the standard behavior
 * for Java object types.
 * <br></br>
 * Implements Serializable since 2.0
 *
 */
class Complex constructor(
    /** The real part.  */
    val real: Double,
    /** The imaginary part.  */
    val imaginary: Double = 0.0
) : FieldElement<Complex> {
    /**
     * Access the imaginary part.
     *
     * @return the imaginary part.
     */
    /**
     * Access the real part.
     *
     * @return the real part.
     */
    /**
     * Checks whether either or both parts of this complex number is
     * `NaN`.
     *
     * @return true if either or both parts of this complex number is
     * `NaN`; false otherwise.
     */
    /** Record whether this complex number is equal to NaN.  */
    val isNaN: Boolean = real.isNaN() || imaginary.isNaN()
    /**
     * Checks whether either the real or imaginary part of this complex number
     * takes an infinite value (either `Double.POSITIVE_INFINITY` or
     * `Double.NEGATIVE_INFINITY`) and neither part
     * is `NaN`.
     *
     * @return true if one or both parts of this complex number are infinite
     * and neither part is `NaN`.
     */
    /** Record whether this complex number is infinite.  */
    val isInfinite: Boolean = !isNaN && (real.isInfinite()) || (imaginary.isInfinite())

    /**
     * Return the absolute value of this complex number.
     * Returns `NaN` if either real or imaginary part is `NaN`
     * and `Double.POSITIVE_INFINITY` if neither part is `NaN`,
     * but at least one part is infinite.
     *
     * @return the absolute value.
     */
    fun abs(): Double {
        if (isNaN) {
            return Double.NaN
        }
        if (isInfinite) {
            return Double.POSITIVE_INFINITY
        }
        return if (abs(real) < abs(imaginary)
        ) {
            if (imaginary == 0.0) {
                return abs(real)
            }
            val q = real / imaginary
            abs(imaginary) * sqrt(1 + q * q)
        } else {
            if (real == 0.0) {
                return abs(imaginary)
            }
            val q = imaginary / real
            abs(real) * sqrt(1 + q * q)
        }
    }

    /**
     * Returns a `Complex` whose value is
     * `(this + addend)`.
     * Uses the definitional formula
     * <pre>
     * `
     * (a + bi) + (c + di) = (a+c) + (b+d)i
    ` *
    </pre> *
     * <br></br>
     * If either `this` or `addend` has a `NaN` value in
     * either part, [.NaN] is returned; otherwise `Infinite`
     * and `NaN` values are returned in the parts of the result
     * according to the rules for [java.lang.Double] arithmetic.
     *
     * @param  a Value to be added to this `Complex`.
     * @return `this + addend`.
     * @throws NullArgumentException if `addend` is `null`.
     */
    override fun add(a: Complex): Complex {
        return if (isNaN || a.isNaN) {
            NaN
        } else createComplex(
            real + a.real,
            imaginary + a.imaginary
        )
    }

    /**
     * Returns a `Complex` whose value is `(this + addend)`,
     * with `addend` interpreted as a real number.
     *
     * @param addend Value to be added to this `Complex`.
     * @return `this + addend`.
     * @see .add
     */
    fun add(addend: Double): Complex {
        return if (isNaN || addend.isNaN()) {
            NaN
        } else createComplex(real + addend, imaginary)
    }

    /**
     * Return the conjugate of this complex number.
     * The conjugate of `a + bi` is `a - bi`.
     * <br></br>
     * [.NaN] is returned if either the real or imaginary
     * part of this Complex number equals `Double.NaN`.
     * <br></br>
     * If the imaginary part is infinite, and the real part is not
     * `NaN`, the returned value has infinite imaginary part
     * of the opposite sign, e.g. the conjugate of
     * `1 + POSITIVE_INFINITY i` is `1 - NEGATIVE_INFINITY i`.
     *
     * @return the conjugate of this Complex object.
     */
    fun conjugate(): Complex {
        return if (isNaN) {
            NaN
        } else createComplex(real, -imaginary)
    }

    /**
     * Returns a `Complex` whose value is
     * `(this / divisor)`.
     * Implements the definitional formula
     * <pre>
     * `
     * a + bi          ac + bd + (bc - ad)i
     * ----------- = -------------------------
     * c + di         c<sup>2</sup> + d<sup>2</sup>
    ` *
    </pre> *
     * but uses
     * [
 * prescaling of operands](http://doi.acm.org/10.1145/1039813.1039814) to limit the effects of overflows and
     * underflows in the computation.
     * <br></br>
     * `Infinite` and `NaN` values are handled according to the
     * following rules, applied in the order presented:
     *
     *  * If either `this` or `divisor` has a `NaN` value
     * in either part, [.NaN] is returned.
     *
     *  * If `divisor` equals [.ZERO], [.NaN] is returned.
     *
     *  * If `this` and `divisor` are both infinite,
     * [.NaN] is returned.
     *
     *  * If `this` is finite (i.e., has no `Infinite` or
     * `NaN` parts) and `divisor` is infinite (one or both parts
     * infinite), [.ZERO] is returned.
     *
     *  * If `this` is infinite and `divisor` is finite,
     * `NaN` values are returned in the parts of the result if the
     * [java.lang.Double] rules applied to the definitional formula
     * force `NaN` results.
     *
     *
     *
     * @param a Value by which this `Complex` is to be divided.
     * @return `this / divisor`.
     * @throws NullArgumentException if `divisor` is `null`.
     */
    override fun divide(a: Complex): Complex {
        if (isNaN || a.isNaN) {
            return NaN
        }
        val c = a.real
        val d = a.imaginary
        if (c == 0.0 && d == 0.0) {
            return NaN
        }
        if (a.isInfinite && !isInfinite) {
            return ZERO
        }
        return if (abs(c) < abs(
                d
            )
        ) {
            val q = c / d
            val denominator = c * q + d
            createComplex(
                (real * q + imaginary) / denominator,
                (imaginary * q - real) / denominator
            )
        } else {
            val q = d / c
            val denominator = d * q + c
            createComplex(
                (imaginary * q + real) / denominator,
                (imaginary - real * q) / denominator
            )
        }
    }

    /**
     * Returns a `Complex` whose value is `(this / divisor)`,
     * with `divisor` interpreted as a real number.
     *
     * @param  divisor Value by which this `Complex` is to be divided.
     * @return `this / divisor`.
     * @see .divide
     */
    fun divide(divisor: Double): Complex {
        if (isNaN || divisor.isNaN()) {
            return NaN
        }
        if (divisor == 0.0) {
            return NaN
        }
        return if (divisor.isInfinite()) {
            if (!isInfinite) ZERO else NaN
        } else createComplex(real / divisor, imaginary / divisor)
    }

    /** {@inheritDoc}  */
    override fun reciprocal(): Complex {
        if (isNaN) {
            return NaN
        }
        if (real == 0.0 && imaginary == 0.0) {
            return INF
        }
        if (isInfinite) {
            return ZERO
        }
        return if (abs(real) < abs(imaginary)) {
            val q = real / imaginary
            val scale = 1.0 / (real * q + imaginary)
            createComplex(scale * q, -scale)
        } else {
            val q = imaginary / real
            val scale = 1.0 / (imaginary * q + real)
            createComplex(scale, -scale * q)
        }
    }

    /**
     * Test for equality with another object.
     * If both the real and imaginary parts of two complex numbers
     * are exactly the same, and neither is `Double.NaN`, the two
     * Complex objects are considered to be equal.
     * The behavior is the same as for JDK's [ Double][Double.equals]:
     *
     *  * All `NaN` values are considered to be equal,
     * i.e, if either (or both) real and imaginary parts of the complex
     * number are equal to `Double.NaN`, the complex number is equal
     * to `NaN`.
     *
     *  *
     * Instances constructed with different representations of zero (i.e.
     * either "0" or "-0") are *not* considered to be equal.
     *
     *
     *
     * @param other Object to test for equality with this instance.
     * @return `true` if the objects are equal, `false` if object
     * is `null`, not an instance of `Complex`, or not equal to
     * this instance.
     */
    override fun equals(other: Any?): Boolean {
        if (this === other) {
            return true
        }
        if (other is Complex) {
            val c = other
            return if (c.isNaN) {
                isNaN
            } else {
                real == c.real && imaginary == c.imaginary
            }
        }
        return false
    }

    /**
     * Get a hashCode for the complex number.
     * Any `Double.NaN` value in real or imaginary part produces
     * the same hash code `7`.
     *
     * @return a hash code value for this object.
     */
    override fun hashCode(): Int {
        return if (isNaN) {
            7
        } else 37 * (17 * imaginary.hashCode() + real.hashCode())
    }

    /**
     * Returns a `Complex` whose value is `this * factor`.
     * Implements preliminary checks for `NaN` and infinity followed by
     * the definitional formula:
     * <pre>
     * `
     * (a + bi)(c + di) = (ac - bd) + (ad + bc)i
    ` *
    </pre> *
     * Returns [.NaN] if either `this` or `factor` has one or
     * more `NaN` parts.
     * <br></br>
     * Returns [.INF] if neither `this` nor `factor` has one
     * or more `NaN` parts and if either `this` or `factor`
     * has one or more infinite parts (same result is returned regardless of
     * the sign of the components).
     * <br></br>
     * Returns finite values in components of the result per the definitional
     * formula in all remaining cases.
     *
     * @param  a value to be multiplied by this `Complex`.
     * @return `this * factor`.
     * @throws NullArgumentException if `factor` is `null`.
     */
    override fun multiply(a: Complex): Complex {
        if (isNaN || a.isNaN) {
            return NaN
        }
        return if (real.isInfinite() || imaginary.isInfinite() ||
            a.real.isInfinite() || a.imaginary.isInfinite()
        ) {
            // we don't use isInfinite() to avoid testing for NaN again
            INF
        } else createComplex(
            real * a.real - imaginary * a.imaginary,
            real * a.imaginary + imaginary * a.real
        )
    }

    /**
     * Returns a `Complex` whose value is `this * factor`, with `factor`
     * interpreted as a integer number.
     *
     * @param  n value to be multiplied by this `Complex`.
     * @return `this * factor`.
     * @see .multiply
     */
    override fun multiply(n: Int): Complex {
        if (isNaN) {
            return NaN
        }
        return if (real.isInfinite() || imaginary.isInfinite()) {
            INF
        } else createComplex(real * n, imaginary * n)
    }

    /**
     * Returns a `Complex` whose value is `this * factor`, with `factor`
     * interpreted as a real number.
     *
     * @param  factor value to be multiplied by this `Complex`.
     * @return `this * factor`.
     * @see .multiply
     */
    fun multiply(factor: Double): Complex {
        if (isNaN || factor.isNaN()) return NaN
        return if (real.isInfinite() || imaginary.isInfinite() || factor.isInfinite()) {
            // we don't use isInfinite() to avoid testing for NaN again
            INF
        } else createComplex(real * factor, imaginary * factor)
    }

    /**
     * Returns a `Complex` whose value is `(-this)`.
     * Returns `NaN` if either real or imaginary
     * part of this Complex number equals `Double.NaN`.
     *
     * @return `-this`.
     */
    override fun negate(): Complex {
        return if (isNaN) {
            NaN
        } else createComplex(-real, -imaginary)
    }

    /**
     * Returns a `Complex` whose value is
     * `(this - subtrahend)`.
     * Uses the definitional formula
     * <pre>
     * `
     * (a + bi) - (c + di) = (a-c) + (b-d)i
    ` *
    </pre> *
     * If either `this` or `subtrahend` has a `NaN]` value in either part,
     * [.NaN] is returned; otherwise infinite and `NaN` values are
     * returned in the parts of the result according to the rules for
     * [java.lang.Double] arithmetic.
     *
     * @param  a value to be subtracted from this `Complex`.
     * @return `this - subtrahend`.
     * @throws NullArgumentException if `subtrahend` is `null`.
     */
    override fun subtract(a: Complex): Complex {
        return if (isNaN || a.isNaN) {
            NaN
        } else createComplex(
            real - a.real,
            imaginary - a.imaginary
        )
    }

    /**
     * Returns a `Complex` whose value is
     * `(this - subtrahend)`.
     *
     * @param  subtrahend value to be subtracted from this `Complex`.
     * @return `this - subtrahend`.
     * @see .subtract
     */
    fun subtract(subtrahend: Double): Complex {
        return if (isNaN || subtrahend.isNaN()) {
            NaN
        } else createComplex(real - subtrahend, imaginary)
    }

    /**
     * Compute the
     * [
 * inverse cosine](http://mathworld.wolfram.com/InverseCosine.html) of this complex number.
     * Implements the formula:
     * <pre>
     * `
     * acos(z) = -i (log(z + i (sqrt(1 - z<sup>2</sup>))))
    ` *
    </pre> *
     * Returns [Complex.NaN] if either real or imaginary part of the
     * input argument is `NaN` or infinite.
     *
     * @return the inverse cosine of this complex number.
     * @since 1.2
     */
    fun acos(): Complex {
        return if (isNaN) {
            NaN
        } else this.add(sqrt1z().multiply(I)).log().multiply(I.negate())
    }

    /**
     * Compute the
     * [
 * inverse sine](http://mathworld.wolfram.com/InverseSine.html) of this complex number.
     * Implements the formula:
     * <pre>
     * `
     * asin(z) = -i (log(sqrt(1 - z<sup>2</sup>) + iz))
    ` *
    </pre> *
     * Returns [Complex.NaN] if either real or imaginary part of the
     * input argument is `NaN` or infinite.
     *
     * @return the inverse sine of this complex number.
     * @since 1.2
     */
    fun asin(): Complex {
        return if (isNaN) {
            NaN
        } else sqrt1z().add(this.multiply(I)).log().multiply(I.negate())
    }

    /**
     * Compute the
     * [
 * inverse tangent](http://mathworld.wolfram.com/InverseTangent.html) of this complex number.
     * Implements the formula:
     * <pre>
     * `
     * atan(z) = (i/2) log((i + z)/(i - z))
    ` *
    </pre> *
     * Returns [Complex.NaN] if either real or imaginary part of the
     * input argument is `NaN` or infinite.
     *
     * @return the inverse tangent of this complex number
     * @since 1.2
     */
    fun atan(): Complex {
        return if (isNaN) {
            NaN
        } else this.add(I).divide(I.subtract(this)).log()
            .multiply(I.divide(createComplex(2.0, 0.0)))
    }

    /**
     * Compute the
     * [
 * cosine](http://mathworld.wolfram.com/Cosine.html)
     * of this complex number.
     * Implements the formula:
     * <pre>
     * `
     * cos(a + bi) = cos(a)cosh(b) - sin(a)sinh(b)i
    ` *
    </pre> *
     * where the (real) functions on the right-hand side are
     * [FastMath.sin], [FastMath.cos],
     * [FastMath.cosh] and [FastMath.sinh].
     * <br></br>
     * Returns [Complex.NaN] if either real or imaginary part of the
     * input argument is `NaN`.
     * <br></br>
     * Infinite values in real or imaginary parts of the input may result in
     * infinite or NaN values returned in parts of the result.
     * <pre>
     * Examples:
     * `
     * cos(1  INFINITY i) = 1 &#x2213; INFINITY i
     * cos(INFINITY + i) = NaN + NaN i
     * cos(INFINITY  INFINITY i) = NaN + NaN i
    ` *
    </pre> *
     *
     * @return the cosine of this complex number.
     * @since 1.2
     */
    fun cos(): Complex {
        return if (isNaN) {
            NaN
        } else createComplex(cos(real) * cosh(imaginary), -sin(real) * sinh(imaginary))
    }

    /**
     * Compute the
     * [
 * hyperbolic cosine](http://mathworld.wolfram.com/HyperbolicCosine.html) of this complex number.
     * Implements the formula:
     * <pre>
     * `
     * cosh(a + bi) = cosh(a)cos(b) + sinh(a)sin(b)i}
    ` *
    </pre> *
     * where the (real) functions on the right-hand side are
     * [FastMath.sin], [FastMath.cos],
     * [FastMath.cosh] and [FastMath.sinh].
     * <br></br>
     * Returns [Complex.NaN] if either real or imaginary part of the
     * input argument is `NaN`.
     * <br></br>
     * Infinite values in real or imaginary parts of the input may result in
     * infinite or NaN values returned in parts of the result.
     * <pre>
     * Examples:
     * `
     * cosh(1  INFINITY i) = NaN + NaN i
     * cosh(INFINITY + i) = INFINITY  INFINITY i
     * cosh(INFINITY  INFINITY i) = NaN + NaN i
    ` *
    </pre> *
     *
     * @return the hyperbolic cosine of this complex number.
     * @since 1.2
     */
    fun cosh(): Complex {
        return if (isNaN) {
            NaN
        } else createComplex(
            cosh(real) * cos(
                imaginary
            ),
            sinh(real) * sin(
                imaginary
            )
        )
    }

    /**
     * Compute the
     * [
 * exponential function](http://mathworld.wolfram.com/ExponentialFunction.html) of this complex number.
     * Implements the formula:
     * <pre>
     * `
     * exp(a + bi) = exp(a)cos(b) + exp(a)sin(b)i
    ` *
    </pre> *
     * where the (real) functions on the right-hand side are
     * [FastMath.exp], [FastMath.cos], and
     * [FastMath.sin].
     * <br></br>
     * Returns [Complex.NaN] if either real or imaginary part of the
     * input argument is `NaN`.
     * <br></br>
     * Infinite values in real or imaginary parts of the input may result in
     * infinite or NaN values returned in parts of the result.
     * <pre>
     * Examples:
     * `
     * exp(1  INFINITY i) = NaN + NaN i
     * exp(INFINITY + i) = INFINITY + INFINITY i
     * exp(-INFINITY + i) = 0 + 0i
     * exp(INFINITY  INFINITY i) = NaN + NaN i
    ` *
    </pre> *
     *
     * @return `*e*<sup>this</sup>`.
     * @since 1.2
     */
    fun exp(): Complex {
        if (isNaN) {
            return NaN
        }
        val expReal: Double = exp(real)
        return createComplex(
            expReal * cos(imaginary),
            expReal * sin(imaginary)
        )
    }

    /**
     * Compute the
     * [
 * natural logarithm](http://mathworld.wolfram.com/NaturalLogarithm.html) of this complex number.
     * Implements the formula:
     * <pre>
     * `
     * log(a + bi) = ln(|a + bi|) + arg(a + bi)i
    ` *
    </pre> *
     * where ln on the right hand side is [FastMath.log],
     * `|a + bi|` is the modulus, [Complex.abs],  and
     * `arg(a + bi) = `[FastMath.atan2](b, a).
     * <br></br>
     * Returns [Complex.NaN] if either real or imaginary part of the
     * input argument is `NaN`.
     * <br></br>
     * Infinite (or critical) values in real or imaginary parts of the input may
     * result in infinite or NaN values returned in parts of the result.
     * <pre>
     * Examples:
     * `
     * log(1  INFINITY i) = INFINITY  (/2)i
     * log(INFINITY + i) = INFINITY + 0i
     * log(-INFINITY + i) = INFINITY + i
     * log(INFINITY  INFINITY i) = INFINITY  (/4)i
     * log(-INFINITY  INFINITY i) = INFINITY  (3/4)i
     * log(0 + 0i) = -INFINITY + 0i
    ` *
    </pre> *
     *
     * @return the value `ln &nbsp; this`, the natural logarithm
     * of `this`.
     * @since 1.2
     */
    fun log(): Complex {
        return if (isNaN) {
            NaN
        } else createComplex(
            ln(abs()),
            atan2(imaginary, real)
        )
    }

    /**
     * Returns of value of this complex number raised to the power of `x`.
     * Implements the formula:
     * <pre>
     * `
     * y<sup>x</sup> = exp(xlog(y))
    ` *
    </pre> *
     * where `exp` and `log` are [.exp] and
     * [.log], respectively.
     * <br></br>
     * Returns [Complex.NaN] if either real or imaginary part of the
     * input argument is `NaN` or infinite, or if `y`
     * equals [Complex.ZERO].
     *
     * @param  x exponent to which this `Complex` is to be raised.
     * @return ` this<sup>`x`</sup>`.
     * @throws NullArgumentException if x is `null`.
     * @since 1.2
     */
    fun pow(x: Complex): Complex {
        return log().multiply(x).exp()
    }

    /**
     * Returns of value of this complex number raised to the power of `x`.
     *
     * @param  x exponent to which this `Complex` is to be raised.
     * @return `this<sup>x</sup>`.
     * @see .pow
     */
    fun pow(x: Double): Complex {
        return log().multiply(x).exp()
    }

    /**
     * Compute the
     * [
 * sine](http://mathworld.wolfram.com/Sine.html)
     * of this complex number.
     * Implements the formula:
     * <pre>
     * `
     * sin(a + bi) = sin(a)cosh(b) - cos(a)sinh(b)i
    ` *
    </pre> *
     * where the (real) functions on the right-hand side are
     * [FastMath.sin], [FastMath.cos],
     * [FastMath.cosh] and [FastMath.sinh].
     * <br></br>
     * Returns [Complex.NaN] if either real or imaginary part of the
     * input argument is `NaN`.
     * <br></br>
     * Infinite values in real or imaginary parts of the input may result in
     * infinite or `NaN` values returned in parts of the result.
     * <pre>
     * Examples:
     * `
     * sin(1  INFINITY i) = 1  INFINITY i
     * sin(INFINITY + i) = NaN + NaN i
     * sin(INFINITY  INFINITY i) = NaN + NaN i
    ` *
    </pre> *
     *
     * @return the sine of this complex number.
     * @since 1.2
     */
    fun sin(): Complex {
        return if (isNaN) {
            NaN
        } else createComplex(
            sin(real) * cosh(
                imaginary
            ),
            cos(real) * sinh(
                imaginary
            )
        )
    }

    /**
     * Compute the
     * [
 * hyperbolic sine](http://mathworld.wolfram.com/HyperbolicSine.html) of this complex number.
     * Implements the formula:
     * <pre>
     * `
     * sinh(a + bi) = sinh(a)cos(b)) + cosh(a)sin(b)i
    ` *
    </pre> *
     * where the (real) functions on the right-hand side are
     * [FastMath.sin], [FastMath.cos],
     * [FastMath.cosh] and [FastMath.sinh].
     * <br></br>
     * Returns [Complex.NaN] if either real or imaginary part of the
     * input argument is `NaN`.
     * <br></br>
     * Infinite values in real or imaginary parts of the input may result in
     * infinite or NaN values returned in parts of the result.
     * <pre>
     * Examples:
     * `
     * sinh(1  INFINITY i) = NaN + NaN i
     * sinh(INFINITY + i) =  INFINITY + INFINITY i
     * sinh(INFINITY  INFINITY i) = NaN + NaN i
    ` *
    </pre> *
     *
     * @return the hyperbolic sine of `this`.
     * @since 1.2
     */
    fun sinh(): Complex {
        return if (isNaN) {
            NaN
        } else createComplex(
            sinh(real) * cos(
                imaginary
            ),
            cosh(real) * sin(
                imaginary
            )
        )
    }

    /**
     * Compute the
     * [
 * square root](http://mathworld.wolfram.com/SquareRoot.html) of this complex number.
     * Implements the following algorithm to compute `sqrt(a + bi)`:
     *  1. Let `t = sqrt((|a| + |a + bi|) / 2)`
     *  1. <pre>if `a &#8805; 0` return `t + (b/2t)i`
     * else return `|b|/2t + sign(b)t i `</pre>
     *
     * where
     *  * `|a| = `[FastMath.abs](a)
     *  * `|a + bi| = `[Complex.abs](a + bi)
     *  * `sign(b) =  `[copySign(1d, b)][FastMath.copySign]
     *
     * <br></br>
     * Returns [Complex.NaN] if either real or imaginary part of the
     * input argument is `NaN`.
     * <br></br>
     * Infinite values in real or imaginary parts of the input may result in
     * infinite or NaN values returned in parts of the result.
     * <pre>
     * Examples:
     * `
     * sqrt(1  INFINITY i) = INFINITY + NaN i
     * sqrt(INFINITY + i) = INFINITY + 0i
     * sqrt(-INFINITY + i) = 0 + INFINITY i
     * sqrt(INFINITY  INFINITY i) = INFINITY + NaN i
     * sqrt(-INFINITY  INFINITY i) = NaN  INFINITY i
    ` *
    </pre> *
     *
     * @return the square root of `this`.
     * @since 1.2
     */
    fun sqrt(): Complex {
        if (isNaN) {
            return NaN
        }
        if (real == 0.0 && imaginary == 0.0) {
            return createComplex(0.0, 0.0)
        }
        val t: Double = sqrt(
            (abs(
                real
            ) + abs()) / 2.0
        )
        return if (real >= 0.0) {
            createComplex(t, imaginary / (2.0 * t))
        } else {
            createComplex(
                abs(imaginary) / (2.0 * t),
                (1.0).withSign(imaginary) * t
            )
        }
    }

    /**
     * Compute the
     * [
 * square root](http://mathworld.wolfram.com/SquareRoot.html) of `1 - this<sup>2</sup>` for this complex
     * number.
     * Computes the result directly as
     * `sqrt(ONE.subtract(z.multiply(z)))`.
     * <br></br>
     * Returns [Complex.NaN] if either real or imaginary part of the
     * input argument is `NaN`.
     * <br></br>
     * Infinite values in real or imaginary parts of the input may result in
     * infinite or NaN values returned in parts of the result.
     *
     * @return the square root of `1 - this<sup>2</sup>`.
     * @since 1.2
     */
    fun sqrt1z(): Complex {
        return createComplex(1.0, 0.0).subtract(this.multiply(this)).sqrt()
    }

    /**
     * Compute the
     * [
 * tangent](http://mathworld.wolfram.com/Tangent.html) of this complex number.
     * Implements the formula:
     * <pre>
     * `
     * tan(a + bi) = sin(2a)/(cos(2a)+cosh(2b)) + [sinh(2b)/(cos(2a)+cosh(2b))]i
    ` *
    </pre> *
     * where the (real) functions on the right-hand side are
     * [FastMath.sin], [FastMath.cos], [FastMath.cosh] and
     * [FastMath.sinh].
     * <br></br>
     * Returns [Complex.NaN] if either real or imaginary part of the
     * input argument is `NaN`.
     * <br></br>
     * Infinite (or critical) values in real or imaginary parts of the input may
     * result in infinite or NaN values returned in parts of the result.
     * <pre>
     * Examples:
     * `
     * tan(a  INFINITY i) = 0  i
     * tan(INFINITY + bi) = NaN + NaN i
     * tan(INFINITY  INFINITY i) = NaN + NaN i
     * tan(/2 + 0 i) = INFINITY + NaN i
    ` *
    </pre> *
     *
     * @return the tangent of `this`.
     * @since 1.2
     */
    fun tan(): Complex {
        if (isNaN || real.isInfinite()) {
            return NaN
        }
        if (imaginary > 20.0) {
            return createComplex(0.0, 1.0)
        }
        if (imaginary < -20.0) {
            return createComplex(0.0, -1.0)
        }
        val real2 = 2.0 * real
        val imaginary2 = 2.0 * imaginary
        val d: Double =
            cos(real2) + cosh(
                imaginary2
            )
        return createComplex(sin(real2) / d, sinh(imaginary2) / d)
    }

    /**
     * Compute the
     * [
 * hyperbolic tangent](http://mathworld.wolfram.com/HyperbolicTangent.html) of this complex number.
     * Implements the formula:
     * <pre>
     * `
     * tan(a + bi) = sinh(2a)/(cosh(2a)+cos(2b)) + [sin(2b)/(cosh(2a)+cos(2b))]i
    ` *
    </pre> *
     * where the (real) functions on the right-hand side are
     * [FastMath.sin], [FastMath.cos], [FastMath.cosh] and
     * [FastMath.sinh].
     * <br></br>
     * Returns [Complex.NaN] if either real or imaginary part of the
     * input argument is `NaN`.
     * <br></br>
     * Infinite values in real or imaginary parts of the input may result in
     * infinite or NaN values returned in parts of the result.
     * <pre>
     * Examples:
     * `
     * tanh(a  INFINITY i) = NaN + NaN i
     * tanh(INFINITY + bi) = 1 + 0 i
     * tanh(INFINITY  INFINITY i) = NaN + NaN i
     * tanh(0 + (/2)i) = NaN + INFINITY i
    ` *
    </pre> *
     *
     * @return the hyperbolic tangent of `this`.
     * @since 1.2
     */
    fun tanh(): Complex {
        if (isNaN || imaginary.isInfinite()) {
            return NaN
        }
        if (real > 20.0) {
            return createComplex(1.0, 0.0)
        }
        if (real < -20.0) {
            return createComplex(-1.0, 0.0)
        }
        val real2 = 2.0 * real
        val imaginary2 = 2.0 * imaginary
        val d: Double =
            cosh(real2) + cos(
                imaginary2
            )
        return createComplex(
            sinh(real2) / d,
            sin(imaginary2) / d
        )
    }

    /**
     * Compute the argument of this complex number.
     * The argument is the angle phi between the positive real axis and
     * the point representing this number in the complex plane.
     * The value returned is between -PI (not inclusive)
     * and PI (inclusive), with negative values returned for numbers with
     * negative imaginary parts.
     * <br></br>
     * If either real or imaginary part (or both) is NaN, NaN is returned.
     * Infinite parts are handled as `Math.atan2` handles them,
     * essentially treating finite parts as zero in the presence of an
     * infinite coordinate and returning a multiple of pi/4 depending on
     * the signs of the infinite parts.
     * See the javadoc for `Math.atan2` for full details.
     *
     * @return the argument of `this`.
     */
    val argument: Double
        get() = atan2(imaginary, real)

    /**
     * Computes the n-th roots of this complex number.
     * The nth roots are defined by the formula:
     * <pre>
     * `
     * z<sub>k</sub> = abs<sup>1/n</sup> (cos(phi + 2k/n) + i (sin(phi + 2k/n))
    ` *
    </pre> *
     * for *`k=0, 1, ..., n-1`*, where `abs` and `phi`
     * are respectively the [modulus][.abs] and
     * [argument][.getArgument] of this complex number.
     * <br></br>
     * If one or both parts of this complex number is NaN, a list with just
     * one element, [.NaN] is returned.
     * if neither part is NaN, but at least one part is infinite, the result
     * is a one-element list containing [.INF].
     *
     * @param n Degree of root.
     * @return a List<Complex> of all `n`-th roots of `this`.
     * @throws NotPositiveException if `n <= 0`.
     * @since 2.0
    </Complex> */
    fun nthRoot(n: Int): List<Complex> {
        if (n <= 0) {
            throw IllegalArgumentException("Non Positive to nthRoot")
        }
        val result: MutableList<Complex> = ArrayList()
        if (isNaN) {
            result.add(NaN)
            return result
        }
        if (isInfinite) {
            result.add(INF)
            return result
        }

        // nth root of abs -- faster / more accurate to use a solver here?
        val nthRootOfAbs: Double = abs().pow(1.0 / n)

        // Compute nth roots of complex number with k = 0, 1, ... n-1
        val nthPhi = argument / n
        val slice: Double = 2 * PI / n
        var innerPart = nthPhi
        for (k in 0 until n) {
            // inner part
            val realPart: Double =
                nthRootOfAbs * cos(innerPart)
            val imaginaryPart: Double =
                nthRootOfAbs * sin(innerPart)
            result.add(createComplex(realPart, imaginaryPart))
            innerPart += slice
        }
        return result
    }

    /**
     * Create a complex number given the real and imaginary parts.
     *
     * @param realPart Real part.
     * @param imaginaryPart Imaginary part.
     * @return a new complex number instance.
     * @since 1.2
     * @see .valueOf
     */
    protected fun createComplex(
        realPart: Double,
        imaginaryPart: Double
    ): Complex {
        return Complex(realPart, imaginaryPart)
    }

    /**
     * Resolve the transient fields in a deserialized Complex Object.
     * Subclasses will need to override [.createComplex] to
     * deserialize properly.
     *
     * @return A Complex instance with all fields resolved.
     * @since 2.0
     */
    protected fun readResolve(): Any {
        return createComplex(real, imaginary)
    }

    /** {@inheritDoc}  */
    override fun toString(): String {
        return "($real, $imaginary)"
    }

    companion object {
        /** The square root of -1. A number representing "0.0 + 1.0i"  */
        val I = Complex(0.0, 1.0)
        // CHECKSTYLE: stop ConstantName
        /** A complex number representing "NaN + NaNi"  */
        val NaN = Complex(Double.NaN, Double.NaN)
        // CHECKSTYLE: resume ConstantName
        /** A complex number representing "+INF + INFi"  */
        val INF = Complex(Double.POSITIVE_INFINITY, Double.POSITIVE_INFINITY)

        /** A complex number representing "1.0 + 0.0i"  */
        val ONE = Complex(1.0, 0.0)

        /** A complex number representing "0.0 + 0.0i"  */
        val ZERO = Complex(0.0, 0.0)

        /** Serializable version identifier  */
        private const val serialVersionUID = -6195664516687396620L
        /**
         * Test for the floating-point equality between Complex objects.
         * It returns `true` if both arguments are equal or within the
         * range of allowed error (inclusive).
         *
         * @param x First value (cannot be `null`).
         * @param y Second value (cannot be `null`).
         * @param maxUlps `(maxUlps - 1)` is the number of floating point
         * values between the real (resp. imaginary) parts of `x` and
         * `y`.
         * @return `true` if there are fewer than `maxUlps` floating
         * point values between the real (resp. imaginary) parts of `x`
         * and `y`.
         *
         * @see Precision.equals
         * @since 3.3
         */
        /**
         * Returns `true` iff the values are equal as defined by
         * [equals(x, y, 1)][.equals].
         *
         * @param x First value (cannot be `null`).
         * @param y Second value (cannot be `null`).
         * @return `true` if the values are equal.
         *
         * @since 3.3
         */
        fun equals(x: Complex, y: Complex, maxUlps: Int = 1): Boolean {
            return org.apache.commons.math4.util.Precision.equals(x.real, y.real, maxUlps) &&
                    org.apache.commons.math4.util.Precision.equals(
                        x.imaginary,
                        y.imaginary,
                        maxUlps
                    )
        }

        /**
         * Returns `true` if, both for the real part and for the imaginary
         * part, there is no double value strictly between the arguments or the
         * difference between them is within the range of allowed error
         * (inclusive).
         *
         * @param x First value (cannot be `null`).
         * @param y Second value (cannot be `null`).
         * @param eps Amount of allowed absolute error.
         * @return `true` if the values are two adjacent floating point
         * numbers or they are within range of each other.
         *
         * @see Precision.equals
         * @since 3.3
         */
        fun equals(x: Complex, y: Complex, eps: Double): Boolean {
            return org.apache.commons.math4.util.Precision.equals(x.real, y.real, eps) &&
                    org.apache.commons.math4.util.Precision.equals(x.imaginary, y.imaginary, eps)
        }

        /**
         * Returns `true` if, both for the real part and for the imaginary
         * part, there is no double value strictly between the arguments or the
         * relative difference between them is smaller or equal to the given
         * tolerance.
         *
         * @param x First value (cannot be `null`).
         * @param y Second value (cannot be `null`).
         * @param eps Amount of allowed relative error.
         * @return `true` if the values are two adjacent floating point
         * numbers or they are within range of each other.
         *
         * @see Precision.equalsWithRelativeTolerance
         * @since 3.3
         */
        fun equalsWithRelativeTolerance(
            x: Complex, y: Complex,
            eps: Double
        ): Boolean {
            return Precision.equalsWithRelativeTolerance(x.real, y.real, eps) &&
                    Precision.equalsWithRelativeTolerance(x.imaginary, y.imaginary, eps)
        }

        /**
         * Create a complex number given the real and imaginary parts.
         *
         * @param realPart Real part.
         * @param imaginaryPart Imaginary part.
         * @return a Complex instance.
         */
        fun valueOf(
            realPart: Double,
            imaginaryPart: Double
        ): Complex {
            return if (realPart.isNaN() || imaginaryPart.isNaN()) {
                NaN
            } else Complex(realPart, imaginaryPart)
        }

        /**
         * Create a complex number given only the real part.
         *
         * @param realPart Real part.
         * @return a Complex instance.
         */
        fun valueOf(realPart: Double): Complex {
            return if (realPart.isNaN()) {
                NaN
            } else Complex(realPart)
        }
    }

}