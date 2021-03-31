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
package org.apache.commons.math4

/**
 * Interface representing [field](http://mathworld.wolfram.com/Field.html) elements.
 * @param <T> the type of the field elements
 * @see Field
 *
 * @since 2.0
</T> */
interface FieldElement<T> {
    /** Compute this + a.
     * @param a element to add
     * @return a new element representing this + a
     *
     */

    fun add(a: T): T

    /** Compute this - a.
     * @param a element to subtract
     * @return a new element representing this - a
     *
     */

    fun subtract(a: T): T

    /**
     * Returns the additive inverse of `this` element.
     * @return the opposite of `this`.
     */
    fun negate(): T

    /** Compute n  this. Multiplication by an integer number is defined
     * as the following sum
     * <center>
     * n  this = <sub>i=1</sub><sup>n</sup> this.
    </center> *
     * @param n Number of times `this` must be added to itself.
     * @return A new element representing n  this.
     */
    fun multiply(n: Int): T

    /** Compute this  a.
     * @param a element to multiply
     * @return a new element representing this  a
     *
     */

    fun multiply(a: T): T

    /** Compute this  a.
     * @param a element to divide by
     * @return a new element representing this  a
     *
     *
     */

    fun divide(a: T): T

    /**
     * Returns the multiplicative inverse of `this` element.
     * @return the inverse of `this`.
     *
     */

    fun reciprocal(): T

}