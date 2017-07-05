/**
 * @file
 * @author Keenan Crane <keenan@cs.caltech.edu>
 * @version 1.0
 *
 * @section LICENSE
 *
 * Copyright 2012 Keenan Crane. All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 * 
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR IMPLIED
 * WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT
 * SHALL THE FREEBSD PROJECT OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
 * OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
 * ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 * 
 * The views and conclusions contained in the software and documentation are those
 * of the author and should not be interpreted as representing official policies,
 * either expressed or implied, of any other person or institution.
 *
 */

#ifndef LIBGEODESIC_HMVECTORSIZET_H
#define LIBGEODESIC_HMVECTORSIZET_H

#include <assert.h>
#include <stddef.h>

/** \brief Variable-sized vector of size_t values.
 *
 * Represents a vector of size_t value which may change in length, stored in
 * a contiguous block of memory.  Elements can be accessed in constant time
 * and appended in amortized constant time.  Accessing entries outside the
 * range 0-(size-1) results in a runtime assertion; this behavior can be
 * disabled by compiling with the flag -DNDEBUG
 *
 */
typedef struct hmVectorSizeT {

   /** \brief Logical length. */
   size_t size;

   /** \brief Amount of physical memory allocated for storage. */
   size_t storage;

   /** \brief Entry values. [1 x storage] */
   size_t* entries;

} hmVectorSizeT;

/** \brief Constructor.
 *
 * Builds an empty vector.
 *
 * @param vector Target object.
 * \memberof hmVectorSizeT
 *
 */
void hmVectorSizeTInitialize( hmVectorSizeT* vector );

/** \brief Destructor.
 *
 * @param vector Target object.
 * \memberof hmVectorSizeT
 *
 */
void hmVectorSizeTDestroy( hmVectorSizeT* vector );

/** \brief Changes the vector length.
 *
 * No guarantees are provided for the entry values of
 * the new vector.
 *
 * @param vector Target object.
 * @param size New size.
 * \memberof hmVectorSizeT
 *
 */
void hmVectorSizeTResize( hmVectorSizeT* vector, size_t size );

/** \brief Sets an entry to the specified value.
 *
 * @param vector Target object.
 * @param index Index in the range 0-(size-1).
 * @param value New value.
 * \memberof hmVectorSizeT
 *
 */
static __inline__ void hmVectorSizeTSetEntry( hmVectorSizeT* vector,
                                              size_t index,
                                              size_t value )
{
   assert( index < vector->size );

   vector->entries[index] = value;
}

/** \brief Gets the value of an entry.
 *
 * @param vector Target object.
 * @param index Index in the range 0-(size-1).
 * @return Entry value.
 * \memberof hmVectorSizeT
 *
 */
static __inline__ size_t hmVectorSizeTGetEntry( hmVectorSizeT* vector,
                                                size_t index )
{
   assert( index < vector->size );

   return vector->entries[index];
}

/** \brief Appends a new element.
 *
 * @param vector Target object.
 * @param value Value of new entry.
 * \memberof hmVectorSizeT
 *
 */
void hmVectorSizeTPushBack( hmVectorSizeT* vector, size_t value );

/** \brief Removes the last element from the end and returns its value.
 *
 * Allows an hmVector to be used as a stack.
 *
 * @param vector Target object.
 * @return value Value of final entry.
 * \memberof hmVectorSizeT
 *
 */
size_t hmVectorSizeTPopBack( hmVectorSizeT* vector );

/** \brief Sorts entries.
 *
 * @param vector Target object.
 * \memberof hmVectorSizeT
 *
 */
void hmVectorSizeTSort( hmVectorSizeT* vector );

#endif // LIBGEODESIC_HMVECTORSIZET_H

