//! Geometric utility functions for distance calculations and spatial operations.

use geo::algorithm::Distance;
use geo::{Euclidean, Line, Point};
use ndarray::Array1;

/// Calculates the minimum distance between a line segment and a circle center.
///
/// # Arguments
///
/// * `line_start` - Starting point of the line segment
/// * `line_end` - Ending point of the line segment
/// * `circle_center` - Center point of the circle
///
/// # Returns
///
/// The minimum Euclidean distance from the circle center to the line segment.
pub fn line_circle_distance(
    line_start: &Array1<f32>,
    line_end: &Array1<f32>,
    circle_center: &Array1<f32>,
) -> f32 {
    let p = Point::new(circle_center[0], circle_center[1]);
    let line = Line::new(
        Point::new(line_start[0], line_start[1]),
        Point::new(line_end[0], line_end[1]),
    );
    Euclidean.distance(&p, &line)
}

/// Wraps a position vector around the simulation box boundaries (toroidal topology).
///
/// # Arguments
///
/// * `v` - Mutable position vector to wrap
/// * `box_width` - Width of the simulation box
/// * `box_height` - Height of the simulation box
pub fn wrap_around_mut(v: &mut Array1<f32>, box_width: f32, box_height: f32) {
    v[0] = v[0].rem_euclid(box_width);
    v[1] = v[1].rem_euclid(box_height);
}
