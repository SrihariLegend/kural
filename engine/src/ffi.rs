//! C interface. Strings are UTF-8 and NUL-terminated; the result is written into a caller's buffer.

use crate::{parse_json, Engine};
use std::ffi::{c_char, CStr};

/// Load a model file. Returns NULL on failure.
#[no_mangle]
pub extern "C" fn kural_open(path: *const c_char) -> *mut Engine {
    if path.is_null() {
        return std::ptr::null_mut();
    }
    let p = unsafe { CStr::from_ptr(path) }.to_string_lossy();
    match Engine::load(&p) {
        Ok(e) => Box::into_raw(Box::new(e)),
        Err(_) => std::ptr::null_mut(),
    }
}

/// Parse `text`. `context` holds the session-context words separated by the byte 0x1f (may be NULL or empty).
/// Writes a JSON array of readings (see parse_json) into `out`. Returns the length of the JSON; if that is
/// >= `cap` nothing useful was written and the call should be repeated with a bigger buffer. -1 on bad input.
#[no_mangle]
pub extern "C" fn kural_parse(e: *const Engine, text: *const c_char, context: *const c_char, beam: i32, keep: i32,
                              out: *mut c_char, cap: usize) -> i64 {
    if e.is_null() || text.is_null() {
        return -1;
    }
    let e = unsafe { &*e };
    let t = unsafe { CStr::from_ptr(text) }.to_string_lossy();
    let ctx: Vec<String> = if context.is_null() {
        Vec::new()
    } else {
        let c = unsafe { CStr::from_ptr(context) }.to_string_lossy();
        if c.is_empty() { Vec::new() } else { c.split('\u{1f}').map(str::to_string).collect() }
    };
    let s = parse_json(e, &t, &ctx, beam.max(1) as usize, keep.max(1) as usize);
    if !out.is_null() && s.len() < cap {
        unsafe {
            std::ptr::copy_nonoverlapping(s.as_ptr(), out as *mut u8, s.len());
            *out.add(s.len()) = 0;
        }
    }
    s.len() as i64
}

#[no_mangle]
pub extern "C" fn kural_close(e: *mut Engine) {
    if !e.is_null() {
        drop(unsafe { Box::from_raw(e) });
    }
}
